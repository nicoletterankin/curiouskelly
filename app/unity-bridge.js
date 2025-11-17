const DEFAULT_EVENTS = [
  'bridge-handshake',
  'session-start',
  'phase-progress',
  'choice-selected',
  'session-complete',
];

const WS_BACKOFF = [1000, 5000, 15000];

export default class UnityBridge {
  constructor(options = {}) {
    this.bridgeVersion = options.bridgeVersion || '1.0.0';
    this.statusEl = document.getElementById('unity-status-label');
    this.postTarget = null;
    this.ws = null;
    this.wsUrl = null;
    this.wsAttempts = 0;
    this.connected = false;

    this.onStatusChange = null;
    this.onConnectionChange = null;
    this.onTelemetry = null;

    window.addEventListener('message', (event) => this.handleMessage(event));
    this.setStatus('Awaiting bridge handshake…');
  }

  setStatus(text) {
    if (this.statusEl) {
      this.statusEl.textContent = text;
    }
    if (typeof this.onStatusChange === 'function') {
      this.onStatusChange(text);
    }
  }

  connectToIframe(targetWindow, origin = '*') {
    if (!targetWindow) return;
    this.postTarget = { targetWindow, origin };
    this.setStatus(`Iframe bridge registered (${origin})`);
    this.notifyConnectionChange('postMessage', 'registered');
  }

  connectWebSocket(url) {
    if (!url || typeof WebSocket === 'undefined') return;
    this.wsUrl = url;
    this.wsAttempts = 0;
    this.openWebSocket();
  }

  openWebSocket() {
    if (!this.wsUrl) return;
    try {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.close();
      }
      this.ws = new WebSocket(this.wsUrl);
    } catch (error) {
      console.warn('[UnityBridge] Failed to initiate WebSocket', error.message);
      return;
    }

    this.ws.onopen = () => {
      this.wsAttempts = 0;
      this.setStatus(`WebSocket connected (${this.wsUrl})`);
      this.notifyConnectionChange('websocket', 'connected');
      this.emit('bridge-handshake', {
        status: 'websocket-connected',
        transport: 'websocket',
        bridgeVersion: this.bridgeVersion,
      });
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'unity-bridge-command') {
          this.routeInboundCommand(data);
        }
      } catch (error) {
        console.warn('[UnityBridge] Invalid WS message', error.message);
      }
    };

    this.ws.onclose = () => {
      this.notifyConnectionChange('websocket', 'disconnected');
      this.setStatus('WebSocket bridge disconnected – retrying…');
      this.scheduleWsReconnect();
    };

    this.ws.onerror = (error) => {
      console.warn('[UnityBridge] WebSocket error', error.message || error);
    };
  }

  scheduleWsReconnect() {
    if (!this.wsUrl) return;
    const delay = WS_BACKOFF[Math.min(this.wsAttempts, WS_BACKOFF.length - 1)];
    this.wsAttempts += 1;
    setTimeout(() => this.openWebSocket(), delay);
  }

  emit(event, payload = {}) {
    const envelope = {
      type: 'unity-bridge-event',
      event,
      payload: {
        ...payload,
        transport: payload.transport || this.getActiveTransport(),
        bridgeVersion: this.bridgeVersion,
      },
      timestamp: new Date().toISOString(),
    };

    if (this.postTarget?.targetWindow && this.postTarget.targetWindow !== window) {
      try {
        this.postTarget.targetWindow.postMessage(envelope, this.postTarget.origin || '*');
      } catch (error) {
        console.warn('[UnityBridge] postMessage failed', error.message);
      }
    }

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(envelope));
    }

    if (!this.postTarget && (!this.ws || this.ws.readyState !== WebSocket.OPEN)) {
      console.debug('[UnityBridge] No active transport; queued event', envelope);
    }
  }

  handleMessage(event) {
    const { data } = event;
    if (!data || typeof data !== 'object') return;
    if (data.type === 'unity-bridge-handshake') {
      this.connected = true;
      this.connectToIframe(event.source, event.origin);
      this.setStatus('Unity bridge connected');
      this.notifyConnectionChange('postMessage', 'connected');
      this.emit('bridge-handshake', { status: 'acknowledged', transport: 'postMessage' });
    } else if (data.type === 'unity-bridge-command') {
      this.routeInboundCommand(data);
    }
  }

  routeInboundCommand(command) {
    switch (command.event) {
      case 'ping':
        this.emit('pong', {});
        break;
      case 'state-update':
        if (typeof this.onTelemetry === 'function') {
          this.onTelemetry(command.payload || {});
        }
        break;
      case 'latency':
        this.setStatus(`Unity latency ${command.payload?.ms ?? '--'} ms`);
        break;
      case 'error':
        this.setStatus(`Unity error: ${command.payload?.message ?? 'unknown'}`);
        break;
      default:
        console.debug('[UnityBridge] inbound', command);
    }
  }

  notifyConnectionChange(channel, state) {
    if (typeof this.onConnectionChange === 'function') {
      this.onConnectionChange(channel, state);
    }
  }

  getActiveTransport() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return 'websocket';
    }
    if (this.postTarget) {
      return 'postMessage';
    }
    return 'none';
  }

  hasActiveTransport() {
    return this.getActiveTransport() !== 'none';
  }

  supportedEvents() {
    return DEFAULT_EVENTS;
  }
}

