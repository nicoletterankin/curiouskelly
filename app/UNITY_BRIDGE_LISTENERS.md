# Unity Bridge Listener Stubs

Reference snippets for Unity teams wiring either WebGL builds (JavaScript) or native desktop clients (C#). Both snippets assume the envelope documented in `UNITY_BRIDGE_PLAN.md`:

```jsonc
{
  "type": "unity-bridge-event",
  "event": "phase-progress",
  "payload": { "sessionId": "...", "phase": "practice" },
  "timestamp": "2025-11-14T06:00:00.000Z"
}
```

## WebGL / `postMessage` Listener (JavaScript)
```js
// Assets/WebGLTemplates/kelly-bridge.js
(function () {
  const parent = window.parent;
  const unityGame = window.unityInstance; // injected by Build.loader.js

  function send(type, event, payload = {}) {
    parent.postMessage(
      {
        type,
        event,
        payload,
        timestamp: new Date().toISOString(),
      },
      '*'
    );
  }

  window.addEventListener('message', (event) => {
    const { data } = event;
    if (!data || data.type !== 'unity-bridge-event') return;

    switch (data.event) {
      case 'session-start':
        unityGame.SendMessage('KellyBridge', 'OnSessionStart', JSON.stringify(data.payload));
        break;
      case 'phase-progress':
        unityGame.SendMessage('KellyBridge', 'OnPhaseProgress', JSON.stringify(data.payload));
        break;
      case 'choice-selected':
        unityGame.SendMessage('KellyBridge', 'OnChoiceSelected', JSON.stringify(data.payload));
        break;
      case 'session-complete':
        unityGame.SendMessage('KellyBridge', 'OnSessionComplete', JSON.stringify(data.payload));
        break;
    }
  });

  // Advertise readiness
  send('unity-bridge-handshake', 'ready', {
    transport: 'postMessage',
    version: '1.0.0',
  });
})();
```

## Unity / iClone Native Client (C# + WebSocket)
```csharp
using System;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public class KellyBridgeClient : MonoBehaviour {
    private ClientWebSocket socket;
    public string bridgeUrl = "ws://localhost:7777";
    public KellyPhaseMapper mapper;

    private async void Start() {
        socket = new ClientWebSocket();
        await socket.ConnectAsync(new Uri(bridgeUrl), CancellationToken.None);
        SendEnvelope("unity-bridge-handshake", new {
            transport = "websocket",
            version = Application.version
        });
        _ = ReceiveLoop();
    }

    private async Task ReceiveLoop() {
        var buffer = new byte[4096];
        while (socket.State == WebSocketState.Open) {
            var result = await socket.ReceiveAsync(buffer, CancellationToken.None);
            if (result.MessageType == WebSocketMessageType.Text) {
                var payload = Encoding.UTF8.GetString(buffer, 0, result.Count);
                HandleEnvelope(payload);
            }
        }
    }

    private void HandleEnvelope(string json) {
        var envelope = JsonUtility.FromJson<BridgeEnvelope>(json);
        switch (envelope.@event) {
            case "phase-progress":
                mapper.PlayPhase(envelope.payload.phase);
                break;
            case "choice-selected":
                mapper.PlayChoiceReaction(envelope.payload.choiceId);
                break;
            case "session-complete":
                mapper.PlayClosingGesture();
                break;
        }
    }

    private async void SendEnvelope(string eventName, object payload) {
        if (socket.State != WebSocketState.Open) return;
        var envelope = new {
            type = "unity-bridge-command",
            event = eventName,
            payload,
            timestamp = DateTime.UtcNow.ToString("o")
        };
        var data = Encoding.UTF8.GetBytes(JsonUtility.ToJson(envelope));
        await socket.SendAsync(data, WebSocketMessageType.Text, true, CancellationToken.None);
    }
}
```

## Animation & Expression Hooks
Both snippets assume a `KellyPhaseMapper` (C#) or similar controller that maps `"welcome"`, `"teaching"`, `"practice"`, `"wisdom"` to animation clips, using the JSON data in `assets/unity/kellyPhaseMap.json`. The mapper should also emit telemetry back to the bridge:

```jsonc
{
  "type": "unity-bridge-command",
  "event": "state-update",
  "payload": {
    "fps": 58,
    "pose": "practice_prompt",
    "latency": 42
  }
}
```

Forward telemetry at ~2 Hz to balance accuracy and bandwidth; include pose IDs, viseme indices, or speaker state as needed for UI badges. 




