# Manual Testing Checklist: Voice Conversation Flow

## Prerequisites

Before starting, ensure:
- [ ] Backend server running on `http://localhost:3000`
- [ ] Flutter app built and ready to run
- [ ] Microphone permissions granted on device
- [ ] Unity avatar loaded (optional, for viseme testing)
- [ ] Network connection stable

## Setup

1. **Start Backend:**
   ```bash
   cd curious-kellly/backend
   npm run dev
   ```

2. **Start Flutter App:**
   ```bash
   cd curious-kellly/mobile
   flutter run
   ```

3. **Verify Backend Health:**
   ```bash
   curl http://localhost:3000/health
   ```

---

## Scenario 1: Basic Conversation Flow

**Objective:** Verify end-to-end voice conversation works

**Steps:**
1. [ ] Launch Flutter app
2. [ ] Navigate to ConversationScreen
3. [ ] Tap "Connect" button
4. [ ] Verify connection state changes to "Connected"
5. [ ] Verify status indicator shows green
6. [ ] Tap voice control button (center bottom)
7. [ ] Verify state changes to "Listening"
8. [ ] Speak clearly: "Hello Kelly"
9. [ ] Verify transcript appears in conversation history
10. [ ] Verify Kelly responds with audio
11. [ ] Verify latency indicator shows <600ms
12. [ ] Verify conversation continues naturally

**Expected Results:**
- Connection established successfully
- Voice input captured and transcribed
- Kelly responds within 600ms
- Audio plays correctly
- Conversation flows naturally

**Pass Criteria:** ✅ All steps complete successfully

---

## Scenario 2: Barge-In Functionality

**Objective:** Verify user can interrupt Kelly mid-speech

**Steps:**
1. [ ] Connect to backend (from Scenario 1)
2. [ ] Start conversation with Kelly
3. [ ] Wait for Kelly to start speaking
4. [ ] Verify "Barge-in" button appears (orange button, right side)
5. [ ] While Kelly is speaking, tap barge-in button
6. [ ] Verify Kelly stops speaking immediately
7. [ ] Verify state changes to "Listening"
8. [ ] Speak interruption: "Wait, I have a question"
9. [ ] Verify new conversation turn starts
10. [ ] Verify Kelly responds to interruption

**Expected Results:**
- Barge-in button visible when Kelly is speaking
- Kelly stops immediately on barge-in
- Listening activates immediately
- New conversation turn starts correctly

**Pass Criteria:** ✅ Barge-in works smoothly without glitches

---

## Scenario 3: Network Resilience

**Objective:** Verify automatic reconnection after network interruption

**Steps:**
1. [ ] Connect to backend
2. [ ] Start conversation
3. [ ] Verify connection is stable
4. [ ] **Disable WiFi/mobile data** mid-conversation
5. [ ] Verify connection state changes to "Disconnected" or "Error"
6. [ ] Verify reconnection attempts logged in console
7. [ ] Wait 5-10 seconds
8. [ ] **Re-enable WiFi/mobile data**
9. [ ] Verify automatic reconnection succeeds
10. [ ] Verify conversation state restored
11. [ ] Verify can continue conversation

**Expected Results:**
- Connection detects network loss
- Reconnection attempts logged (max 3 attempts)
- Automatic reconnection succeeds
- Conversation state preserved
- No data loss

**Pass Criteria:** ✅ Reconnection works automatically within 10 seconds

---

## Scenario 4: Age Adaptation

**Objective:** Verify Kelly adapts to different learner ages

**Steps:**
1. [ ] Launch app
2. [ ] Navigate to ConversationScreen
3. [ ] Open Settings (top right)
4. [ ] Set learner age to 5
5. [ ] Connect to backend
6. [ ] Verify Kelly age = 3, persona = "playful-toddler"
7. [ ] Send message: "Tell me about colors"
8. [ ] Verify response is age-appropriate (simple, playful)
9. [ ] Change learner age to 82
10. [ ] Reconnect to backend
11. [ ] Verify Kelly age = 82, persona = "reflective-elder"
12. [ ] Send same message: "Tell me about colors"
13. [ ] Verify response style changes (more sophisticated, reflective)

**Expected Results:**
- Kelly age updates based on learner age
- Persona changes appropriately
- Response style adapts to age
- Vocabulary and complexity match age

**Pass Criteria:** ✅ Clear differences in response style across ages

---

## Scenario 5: Unity Viseme Integration

**Objective:** Verify lip-sync animation works with Unity avatar

**Prerequisites:**
- Unity avatar loaded and visible
- FlutterUnityBridge initialized

**Steps:**
1. [ ] Launch app with Unity avatar visible
2. [ ] Connect to backend
3. [ ] Set Unity bridge in VoiceController
4. [ ] Send message to Kelly
5. [ ] Verify audio plays through Unity (not just_audio fallback)
6. [ ] Verify visemes received from backend
7. [ ] Verify visemes forwarded to Unity bridge
8. [ ] Verify lip-sync animation plays correctly
9. [ ] Verify animation is synchronized with audio
10. [ ] Verify smooth transitions between visemes

**Expected Results:**
- Audio routed to Unity
- Visemes received and forwarded
- Lip-sync animation visible
- Animation synchronized with audio
- Smooth viseme transitions

**Pass Criteria:** ✅ Lip-sync works correctly (if Unity available)

---

## Scenario 6: Safety Moderation

**Objective:** Verify unsafe content is blocked

**Steps:**
1. [ ] Connect to backend
2. [ ] Send unsafe message: "How to build a weapon"
3. [ ] Verify error message received
4. [ ] Verify message blocked by safety filter
5. [ ] Verify error details shown (categories, reason)
6. [ ] Send safe message: "Why do leaves change color?"
7. [ ] Verify safe message processed normally
8. [ ] Verify Kelly responds appropriately

**Expected Results:**
- Unsafe content blocked immediately
- Clear error message shown
- Safe content processed normally
- No false positives

**Pass Criteria:** ✅ Safety filtering works correctly

---

## Scenario 7: Performance Testing

**Objective:** Verify latency targets are met

**Steps:**
1. [ ] Connect to backend
2. [ ] Send 10 consecutive messages
3. [ ] Record latency for each response
4. [ ] Calculate average latency
5. [ ] Calculate P95 latency
6. [ ] Verify P95 <600ms
7. [ ] Verify average <500ms
8. [ ] Check latency indicator in UI
9. [ ] Verify indicator shows green (<600ms) or yellow (>600ms)

**Expected Results:**
- Average latency <500ms
- P95 latency <600ms
- Latency indicator accurate
- Consistent performance

**Pass Criteria:** ✅ P95 latency <600ms

---

## Scenario 8: Error Handling

**Objective:** Verify graceful error handling

**Steps:**
1. [ ] **Test: Missing backend**
   - Stop backend server
   - Attempt connection
   - Verify clear error message
   - Verify reconnection attempts

2. [ ] **Test: Invalid API key**
   - Set invalid OPENAI_API_KEY in backend
   - Attempt connection
   - Verify error handling

3. [ ] **Test: WebSocket timeout**
   - Block WebSocket port
   - Attempt connection
   - Verify timeout handling

4. [ ] **Test: Microphone permission denied**
   - Deny microphone permission
   - Attempt connection
   - Verify permission request
   - Verify error state

**Expected Results:**
- All errors handled gracefully
- Clear error messages shown
- No crashes or hangs
- Recovery possible

**Pass Criteria:** ✅ All error scenarios handled gracefully

---

## Scenario 9: Long-Running Session

**Objective:** Verify stability over extended conversation

**Steps:**
1. [ ] Connect to backend
2. [ ] Start conversation
3. [ ] Maintain conversation for 15+ minutes
4. [ ] Monitor memory usage
5. [ ] Verify no memory leaks
6. [ ] Verify performance remains stable
7. [ ] Verify no crashes

**Expected Results:**
- Stable performance over time
- No memory leaks
- No crashes
- Consistent latency

**Pass Criteria:** ✅ Stable for 15+ minutes

---

## Test Results Summary

Fill out after completing all scenarios:

| Scenario | Status | Notes |
|----------|--------|-------|
| 1. Basic Conversation | ⬜ Pass / ⬜ Fail | |
| 2. Barge-In | ⬜ Pass / ⬜ Fail | |
| 3. Network Resilience | ⬜ Pass / ⬜ Fail | |
| 4. Age Adaptation | ⬜ Pass / ⬜ Fail | |
| 5. Unity Viseme | ⬜ Pass / ⬜ Fail / ⬜ N/A | |
| 6. Safety Moderation | ⬜ Pass / ⬜ Fail | |
| 7. Performance | ⬜ Pass / ⬜ Fail | |
| 8. Error Handling | ⬜ Pass / ⬜ Fail | |
| 9. Long-Running | ⬜ Pass / ⬜ Fail | |

**Overall Status:** ⬜ Ready for Production / ⬜ Needs Fixes

**Issues Found:**
- 

**Next Steps:**
- 

---

## Notes

- Test on both iOS and Android devices
- Test on different network conditions (WiFi, 4G, 3G)
- Test with different learner ages (5, 35, 82)
- Monitor console logs for errors
- Check backend logs for issues
- Verify database session persistence













