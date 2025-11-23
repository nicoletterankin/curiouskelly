# Unity Bridge End-to-End Test Matrix

| ID | Scenario | Steps | Expected |
|----|----------|-------|----------|
| T1 | Session API online, iframe bridge | 1. Start backend (`pnpm dev`).<br>2. `python -m http.server 8000` and open `/app/index.html`.<br>3. Set `data-src` on `#unity-iframe` to a test page that posts the handshake. | Status badge flips to “Unity bridge connected”, overlay hides, `session-start` / `phase-progress` emit to console. |
| T2 | Session API offline, iframe bridge | 1. Stop backend.<br>2. Reload app with handshake page still active. | Lesson still plays, “Offline mode” message appears, Unity bridge remains connected. |
| T3 | WebSocket relay happy path | 1. Run local relay `ws://localhost:7777` that echoes `unity-bridge-command` frames.<br>2. Open app; ensure `data-ws-url` points to relay.<br>3. Relay replies with `state-update`. | Badge displays `Streaming 60 fps • pose-name`, overlay hidden, `session-start` events appear in relay logs. |
| T4 | WebSocket reconnect | 1. With T3 running, drop the relay connection.<br>2. Observe console + badge.<br>3. Restart relay within 15s. | Badge shows “WebSocket bridge disconnected – retrying…” then “WebSocket connected …”; events resume without page refresh. |
| T5 | Telemetry + errors | 1. Unity sends `state-update` every 500 ms and an `error` event when animation missing.<br>2. Trigger missing clip by selecting unsupported choice. | Badge rotates between streaming info and “Unity error: …”; console logs inbound command. |
| T6 | Choice → animation mapping | 1. Import `assets/unity/kellyPhaseMap.json` into Unity mapper.<br>2. For each phase/choice, confirm matching animation is played (visual check). | Each event references valid entry; telemetry matches active animation ID. |
| T7 | Wisdom completion | 1. Finish a lesson so the web shell calls `/complete` and emits `session-complete`.<br>2. Unity responds with closing pose + `state-update`. | Badge shows celebration text, session streak increments, Unity transitions to `Kelly_Wisdom_Celebrate`. |
| T8 | Security / origin | 1. Set `data-target-origin` to a specific origin and try posting handshake from another origin. | Handshake ignored; console warning appears. |

## Execution Notes
- Keep DevTools console open to verify outbound envelopes (`unityBridge.emit` logs when no transport available).  
- For iframe tests, a simple `handshake.html` page that posts `unity-bridge-handshake` is sufficient.  
- For WebSocket tests, `wscat` or a tiny Node relay can mimic Unity by replying with JSON.
- Record FPS/latency from telemetry payloads to ensure the UI badge surfaces the latest values.  
- When filing bugs, include the envelope JSON plus the exact transport (postMessage / websocket) observed. 








