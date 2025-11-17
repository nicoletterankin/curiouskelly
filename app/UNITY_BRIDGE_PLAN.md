# Unity / iClone Bridge Plan (Phase 4)

**Status:** Planning + Frontend Stubs Ready  
**Goal:** Drive Kelly’s 3D avatar (Unity WebGL or native client) with the same session + phase data used by the VisionOS web shell, while receiving playback / pose acknowledgements back from Unity.

---

## 1. Architecture Overview

1. **Web Shell (this repo / `app/`)**
   - Emits semantic events (`session-start`, `phase-progress`, `choice-selected`, `session-complete`) through `UnityBridge`.
   - Listens for handshake + optional commands from Unity (e.g., ping, availability, pause/state updates).
2. **Transport Layer**
   - **Primary:** `window.postMessage` between the shell and an embedded Unity WebGL canvas (iframe or same page).  
   - **Fallback / Native Client:** Secure WebSocket to localhost (or LAN) daemon that proxies to an iClone/Unity runtime. Same command schema applies.
3. **Unity/iClone Runtime**
   - Subscribes to commands, plays the correct animation/audio, and optionally emits telemetry (latency, viseme state, completion).
   - Sends `unity-bridge-handshake` on load, then `unity-bridge-command` messages for runtime events.

```
Web Shell ── UnityBridge (JS) ── postMessage / WS ── Unity WebGL / Native ── iClone Avatar
```

---

## 2. Event & Command Schema

All envelopes share the same top-level structure:

```jsonc
{
  "type": "unity-bridge-event",        // or unity-bridge-command (inbound)
  "event": "session-start",
  "payload": { ... },
  "timestamp": "2025-11-14T05:00:00.000Z"
}
```

### Outbound (Web ➜ Unity)
| Event | Purpose | Payload |
|-------|---------|---------|
| `bridge-handshake` | Sent automatically when Unity acknowledges connection. | `{ status: "acknowledged" }` |
| `session-start` | Fired after API session start or resume. | `{ mode: "new"|"resume", sessionId, lessonId, phase }` |
| `phase-progress` | Fired whenever the current phase changes or progress is saved. | `{ sessionId, phase, completedPhase? }` |
| `choice-selected` | Fired when a learner taps a choice card. | `{ sessionId, currentPhase, nextPhase, choiceId }` |
| `session-complete` | Fired once the wisdom phase auto-completes the backend session. | `{ lessonId, durationMin }` |

### Inbound (Unity ➜ Web)
| Event | Purpose | Expected Action |
|-------|---------|-----------------|
| `unity-bridge-handshake` | Announces Unity is ready. Contains `transport`, `version`. | UnityBridge stores target window/origin and replies with `bridge-handshake`. |
| `ping` | Health check from Unity. | UnityBridge replies with `pong`. |
| `state-update` *(future)* | Reports avatar pose, viseme index, or audio markers. | Update UI / analytics. |
| `request-pause` *(future)* | Ask the shell to pause the lesson/audio. | Hook into audio controller & session pause endpoint. |
| `error` *(future)* | Unity surfaced fault. | Show user-facing warning + log. |

#### Transport-Specific Meta

- **postMessage**: envelopes include `payload.transport = "postMessage"` plus `origin` recorded from the iframe.  
- **WebSocket**: envelopes include `payload.transport = "websocket"` and the relay URL (e.g., `ws://localhost:7777`). Unity should reply with `{ type: "unity-bridge-handshake", event: "ready", transport: "websocket", version: "1.0.0" }`.

---

## 3. Handshake & Message Flow

1. Unity WebGL loads and immediately posts `{ type: "unity-bridge-handshake", event: "ready", version: "1.0" }` to the parent window.
2. `UnityBridge` stores the source/origin, updates the UI badge (“Unity bridge connected”), and emits `bridge-handshake` back.
3. As soon as the learner loads or resumes a lesson, the shell emits `session-start`. Unity can then preload mouth/pose clips for the current phase.
4. For each phase or learner interaction, the shell fires `phase-progress` / `choice-selected`. Unity uses this to select the appropriate animation, lip-sync file, or camera move.
5. When the learner reaches Wisdom, the shell marks the backend session complete and emits `session-complete`. Unity can transition Kelly to a closing pose.

---

## 4. Data Contracts

```jsonc
// session-start payload
{
  "mode": "resume",
  "sessionId": "b0aa...",
  "lessonId": "the-sun",
  "phase": "teaching"
}

// phase-progress payload
{
  "sessionId": "b0aa...",
  "phase": "practice",
  "completedPhase": "teaching"
}

// choice-selected payload
{
  "sessionId": "b0aa...",
  "currentPhase": "practice",
  "nextPhase": "wisdom",
  "choiceId": "understanding_the_sun_reveals_fundamental_principles"
}
```

For inbound telemetry we’ll mirror the same shape but under `unity-bridge-command`:

```jsonc
{
  "type": "unity-bridge-command",
  "event": "state-update",
  "payload": {
    "viseme": 14,
    "pose": "listening_positive",
    "confidence": 0.92
  }
}
```

---

## 5. Embedding & Transport Options

| Mode | Usage | Notes |
|------|-------|-------|
| **Inline WebGL** | Ideal for browsers on desktop. Unity canvas sits inside the center column, and `postMessage` stays in-page. | Must ensure iframe/webgl uses the same origin or properly configured `targetOrigin`. |
| **Local Native Client** | For studios running Kelly on a dedicated render machine. Web shell talks to `ws://localhost:7777` (Electron or .NET relay) which forwards commands to Unity/iClone. | UnityBridge will need a WebSocket transport variant—schema identical. |
| **Cloud Stream (future)** | Web sends commands to a cloud-hosted Unity session (e.g., Pixel Streaming). | Requires auth + relay tokens; out of scope for Phase 4 but schema-ready. |

### Configuration Surface

| Attribute | Location | Description |
|-----------|----------|-------------|
| `data-src` | `#unity-iframe` | Unity WebGL build URL. Empty keeps iframe hidden. |
| `data-target-origin` | `#unity-iframe` | Expected origin for postMessage validation (e.g., `https://unity.local`). |
| `data-ws-url` | `#unity-iframe` | Local relay endpoint (e.g., `ws://localhost:7777`). |
| `window.UNITY_BRIDGE_CONFIG` *(optional)* | Global | Override defaults, provide auth tokens, or disable transports. |

When the iframe finishes loading, `UnityBridge.connectToIframe(iframe.contentWindow, origin)` runs automatically. For WS, `UnityBridge.connectWebSocket(url)` retries with exponential backoff (1s, 5s, 15s).

---

## 6. Next Implementation Steps

1. **Activate the transport** (choose inline WebGL iframe or WebSocket). Pass the iframe window to `UnityBridge.connect(...)` after it loads.
2. **Unity listener** subscribes to `message` events (WebGL) or WS data, validates the `unity-bridge-event`, and routes to animation controllers.
3. **iClone command mapping** – map each `phase` to the correct iClone animation kit (idle, speak, thoughtful, celebrate). Provide a JSON file that pairs `lessonId` + `phase` with asset bundle IDs.
4. **Telemetry loop** – once Unity can emit `state-update`, light up the on-screen badge with “Streaming @60fps” or error states.
5. **Testing Checklist** – confirm handshake, session start, phase transitions, and completion signals while backend sessions are active (Phase 3 stack already online).

This plan keeps the bridge schema versioned and transport-agnostic so the same events can drive WebGL, native Unity, or even iClone command servers without refactoring the lesson player. Once the Unity team confirms the receiver implementation, we can switch from console logging to real-time avatar control by simply wiring the iframe/WS endpoint. 

