/**
 * Kelly Pixi Compositor (WebGL overlay layer)
 * - Renders a procedural mouth + blink overlay on top of the HeyGen full-frame video (white background).
 * - Uses static face anchors (demo-safe). No segmentation required.
 *
 * Requirements:
 * - pixi.js available as global `PIXI` (loaded via CDN in learn.html)
 *
 * API:
 * - KellyPixiCompositor.init({ containerEl, width, height })
 * - KellyPixiCompositor.attachVideo(videoEl)
 * - KellyPixiCompositor.setFaceAnchor({ x, y, scale, rotation })
 * - KellyPixiCompositor.setBlendshapes(blendshapes)
 * - KellyPixiCompositor.setEnabled(true/false)
 *
 * Debug:
 * - Add `?pixiDebug=1` to render a red anchor dot so you can visually confirm overlays are rendering.
 */
(() => {
  const DEFAULT_ANCHOR = {
    // Normalized coordinates (0..1) in the video frame
    // Tuned for `public/kelly/videos/001/welcome.mp4` (white background talking head)
    x: 0.5,
    y: 0.54,
    scale: 1.0,
    rotation: 0,
  };

  function clamp01(n) {
    const v = Number(n);
    if (!Number.isFinite(v)) return 0;
    return Math.max(0, Math.min(1, v));
  }

  function clamp(n, a, b) {
    const v = Number(n);
    if (!Number.isFinite(v)) return a;
    return Math.max(a, Math.min(b, v));
  }

  const KellyPixiCompositor = {
    isInitialized: false,
    isEnabled: true,
    containerEl: null,
    videoEl: null,
    app: null,
    overlayRoot: null,
    mouth: null,
    upperLip: null,
    lowerLip: null,
    teeth: null,
    blinkLeft: null,
    blinkRight: null,
    anchor: { ...DEFAULT_ANCHOR },
    lastBlendshapes: {},
    _blinkTimer: 0,
    _blinkState: 0,
    _blinkPhase: 0,
    _debugMarker: null,

    init(options = {}) {
      if (this.isInitialized) return this;
      if (typeof window === 'undefined' || !window.PIXI) {
        console.warn('[KellyPixiCompositor] PIXI not found; compositor disabled');
        return this;
      }

      this.containerEl =
        options.containerEl ||
        (typeof document !== 'undefined' ? document.getElementById('kelly-stage') : null);

      if (!this.containerEl) {
        console.warn('[KellyPixiCompositor] container not found');
        return this;
      }

      const rect = this.containerEl.getBoundingClientRect();
      const width = Math.max(1, Math.floor(options.width || rect.width || 720));
      const height = Math.max(1, Math.floor(options.height || rect.height || 1280));

      this.app = new window.PIXI.Application({
        width,
        height,
        backgroundAlpha: 0,
        antialias: true,
        autoDensity: true,
        resolution: Math.min(2, window.devicePixelRatio || 1),
      });

      // Put canvas on top of the video (pointer-events none)
      const view = this.app.view;
      view.style.position = 'absolute';
      view.style.inset = '0';
      view.style.width = '100%';
      view.style.height = '100%';
      view.style.pointerEvents = 'none';
      view.style.zIndex = '20';

      // Ensure container is positioned
      const cs = window.getComputedStyle(this.containerEl);
      if (cs.position === 'static') {
        this.containerEl.style.position = 'relative';
      }
      this.containerEl.appendChild(view);

      this.overlayRoot = new window.PIXI.Container();
      this.app.stage.addChild(this.overlayRoot);

      this._buildOverlays();

      // Animation loop
      this.app.ticker.add((delta) => {
        this._tick(delta);
      });

      // Resize observer: keep canvas sized to container
      try {
        const ro = new ResizeObserver(() => this._resizeToContainer());
        ro.observe(this.containerEl);
        this._resizeObserver = ro;
      } catch (_) {}

      this.isInitialized = true;
      // Use warn so it’s visible in demo console logs.
      console.warn('[KellyPixiCompositor] Initialized');
      try { window.__KELLY_PIXI_READY = true; } catch (_) {}
      return this;
    },

    attachVideo(videoEl) {
      this.videoEl = videoEl || null;
      return this;
    },

    setEnabled(enabled) {
      this.isEnabled = !!enabled;
      if (this.app?.view) this.app.view.style.display = this.isEnabled ? '' : 'none';
      return this;
    },

    setFaceAnchor(anchor) {
      if (!anchor) return this;
      this.anchor = {
        x: clamp01(anchor.x ?? this.anchor.x),
        y: clamp01(anchor.y ?? this.anchor.y),
        scale: clamp(anchor.scale ?? this.anchor.scale, 0.5, 2.0),
        rotation: clamp(anchor.rotation ?? this.anchor.rotation, -0.5, 0.5),
      };
      return this;
    },

    setBlendshapes(blendshapes) {
      this.lastBlendshapes = blendshapes || {};
      return this;
    },

    _resizeToContainer() {
      if (!this.app || !this.containerEl) return;
      const rect = this.containerEl.getBoundingClientRect();
      const w = Math.max(1, Math.floor(rect.width));
      const h = Math.max(1, Math.floor(rect.height));
      if (w === this.app.renderer.width && h === this.app.renderer.height) return;
      this.app.renderer.resize(w, h);
    },

    _buildOverlays() {
      // Mouth group (procedural)
      const mouth = new window.PIXI.Container();
      mouth.name = 'mouth';
      this.mouth = mouth;

      // Mouth interior
      const mouthInterior = new window.PIXI.Graphics();
      mouthInterior.name = 'mouthInterior';

      // Lips
      const upperLip = new window.PIXI.Graphics();
      upperLip.name = 'upperLip';
      const lowerLip = new window.PIXI.Graphics();
      lowerLip.name = 'lowerLip';
      this.upperLip = upperLip;
      this.lowerLip = lowerLip;

      // Teeth highlight (subtle)
      const teeth = new window.PIXI.Graphics();
      teeth.name = 'teeth';
      this.teeth = teeth;

      mouth.addChild(mouthInterior);
      mouth.addChild(teeth);
      mouth.addChild(upperLip);
      mouth.addChild(lowerLip);

      this._mouthInterior = mouthInterior;

      // Blink overlays (subtle eyelid sweeps)
      const blinkLeft = new window.PIXI.Graphics();
      const blinkRight = new window.PIXI.Graphics();
      blinkLeft.name = 'blinkLeft';
      blinkRight.name = 'blinkRight';
      this.blinkLeft = blinkLeft;
      this.blinkRight = blinkRight;

      this.overlayRoot.addChild(blinkLeft);
      this.overlayRoot.addChild(blinkRight);
      this.overlayRoot.addChild(mouth);

      // Optional debug marker to prove overlay is rendering (opt-in)
      try {
        const isDebug = (typeof location !== 'undefined') && location.search.includes('pixiDebug=1');
        if (isDebug) {
          const marker = new window.PIXI.Graphics();
          marker.name = 'debugMarker';
          marker.beginFill(0xff3b30, 0.9);
          marker.drawCircle(0, 0, 10);
          marker.endFill();
          this._debugMarker = marker;
          this.overlayRoot.addChild(marker);
        }
      } catch (_) {}
    },

    _tick(delta) {
      if (!this.isEnabled || !this.app) return;
      this._updateBlink(delta);
      this._renderOverlaysFromBlendshapes(this.lastBlendshapes || {});
    },

    _updateBlink(delta) {
      // Simple deterministic blink loop
      // - Every ~4-6 seconds blink once (fast close + open)
      const dt = (delta || 1) / 60;
      this._blinkTimer += dt;

      if (this._blinkState === 0) {
        const next = 4.0 + (Math.sin(this._blinkTimer * 0.7) * 1.0 + 1.0) * 1.0; // ~4-6s
        if (this._blinkTimer > next) {
          this._blinkState = 1;
          this._blinkPhase = 0;
        }
      } else {
        this._blinkPhase += dt;
        if (this._blinkPhase >= 0.16) {
          this._blinkState = 0;
          this._blinkTimer = 0;
        }
      }
    },

    _getBlinkAmount() {
      if (this._blinkState === 0) return 0;
      // ease in/out over 0.16s
      const t = clamp(this._blinkPhase / 0.16, 0, 1);
      const eased = t < 0.5 ? (t * 2) : (2 - t * 2);
      return clamp(eased, 0, 1);
    },

    _renderOverlaysFromBlendshapes(bs) {
      const r = this.app.renderer;
      if (!r) return;

      // Anchor in pixels
      const ax = this.anchor.x * r.width;
      const ay = this.anchor.y * r.height;
      const s = this.anchor.scale;

      // Convert blendshapes (0..100-ish) into mouth params
      const jawOpen = clamp((bs.jawOpen ?? bs.mouthOpen ?? 0) / 100, 0, 1);
      const funnel = clamp((bs.mouthFunnel ?? 0) / 100, 0, 1);
      const pucker = clamp((bs.mouthPucker ?? 0) / 100, 0, 1);
      const stretch = clamp(((bs.mouthStretchLeft ?? 0) + (bs.mouthStretchRight ?? 0)) / 200, 0, 1);

      // Mouth size in pixels (tuned visually)
      const baseW = 150 * s;
      const baseH = 60 * s;
      const openH = baseH + jawOpen * 120 * s;
      const w = baseW + stretch * 80 * s - pucker * 40 * s;
      const h = Math.max(12 * s, openH);

      // Mouth position offsets relative to anchor
      const mx = ax;
      const my = ay + 80 * s; // mouth sits below the anchor center

      // Draw mouth interior
      const mouthInterior = this._mouthInterior;
      mouthInterior.clear();
      mouthInterior.beginFill(0x5b0c0c, 0.88);
      mouthInterior.drawRoundedRect(-w / 2, -h / 2, w, h, 18 * s);
      mouthInterior.endFill();

      // Teeth (only when somewhat open)
      this.teeth.clear();
      if (jawOpen > 0.18) {
        this.teeth.beginFill(0xffffff, 0.22);
        this.teeth.drawRoundedRect(-w * 0.28, -h * 0.35, w * 0.56, h * 0.18, 8 * s);
        this.teeth.endFill();
      }

      // Lips (subtle)
      const lipAlpha = 0.28 + funnel * 0.12;
      const lipColor = 0xa86b6b;
      const upper = this.upperLip;
      upper.clear();
      upper.beginFill(lipColor, lipAlpha);
      upper.drawRoundedRect(-w / 2, -h / 2 - 10 * s, w, 18 * s, 10 * s);
      upper.endFill();

      const lower = this.lowerLip;
      lower.clear();
      lower.beginFill(lipColor, lipAlpha);
      lower.drawRoundedRect(-w / 2, h / 2 - 8 * s, w, 18 * s, 10 * s);
      lower.endFill();

      // Place mouth container
      this.mouth.x = mx;
      this.mouth.y = my;
      this.mouth.rotation = this.anchor.rotation || 0;

      if (this._debugMarker) {
        this._debugMarker.x = ax;
        this._debugMarker.y = ay;
      }

      // Blink overlays: draw two semi-transparent eyelid bands
      const blink = this._getBlinkAmount();
      const blinkOpacity = 0.18 + blink * 0.6;
      const eyelidH = (4 + blink * 40) * s;
      const eyeW = 120 * s;

      // Left/right eye positions relative to anchor
      const eyeY = ay - 10 * s;
      const eyeDX = 110 * s;

      this.blinkLeft.clear();
      this.blinkRight.clear();
      if (blink > 0.01) {
        this.blinkLeft.beginFill(0xffffff, blinkOpacity);
        this.blinkLeft.drawRoundedRect(ax - eyeDX - eyeW / 2, eyeY - eyelidH / 2, eyeW, eyelidH, 10 * s);
        this.blinkLeft.endFill();

        this.blinkRight.beginFill(0xffffff, blinkOpacity);
        this.blinkRight.drawRoundedRect(ax + eyeDX - eyeW / 2, eyeY - eyelidH / 2, eyeW, eyelidH, 10 * s);
        this.blinkRight.endFill();
      }
    },
  };

  window.KellyPixiCompositor = KellyPixiCompositor;
})();

/**
 * Kelly Pixi Compositor (WebGL overlay layer)
 * - Renders a procedural mouth + blink overlay on top of the HeyGen full-frame video (white background).
 * - Uses static face anchors (demo-safe). No segmentation required.
 *
 * Requirements:
 * - pixi.js available as global `PIXI` (loaded via CDN in learn.html)
 *
 * API:
 * - KellyPixiCompositor.init({ containerEl, width, height })
 * - KellyPixiCompositor.attachVideo(videoEl)
 * - KellyPixiCompositor.setFaceAnchor({ x, y, scale, rotation })
 * - KellyPixiCompositor.setBlendshapes(blendshapes)
 * - KellyPixiCompositor.setEnabled(true/false)
 */

(function () {
  const DEFAULT_ANCHOR = {
    // Normalized coordinates (0..1) in the video frame
    // Tuned for `public/kelly/videos/001/welcome.mp4` (white background talking head)
    x: 0.5,
    y: 0.54,
    scale: 1.0,
    rotation: 0,
  };

  function clamp01(n) {
    const v = Number(n);
    if (!Number.isFinite(v)) return 0;
    return Math.max(0, Math.min(1, v));
  }

  function clamp(n, a, b) {
    const v = Number(n);
    if (!Number.isFinite(v)) return a;
    return Math.max(a, Math.min(b, v));
  }

  const KellyPixiCompositor = {
    isInitialized: false,
    isEnabled: true,
    containerEl: null,
    videoEl: null,
    app: null,
    overlayRoot: null,
    mouth: null,
    upperLip: null,
    lowerLip: null,
    teeth: null,
    blinkLeft: null,
    blinkRight: null,
    anchor: { ...DEFAULT_ANCHOR },
    lastBlendshapes: {},
    _blinkTimer: 0,
    _blinkState: 0,

    init(options = {}) {
      if (this.isInitialized) return this;
      if (typeof window === 'undefined' || !window.PIXI) {
        console.warn('[KellyPixiCompositor] PIXI not found; compositor disabled');
        return this;
      }

      this.containerEl =
        options.containerEl ||
        (typeof document !== 'undefined' ? document.getElementById('kelly-stage') : null);

      if (!this.containerEl) {
        console.warn('[KellyPixiCompositor] container not found');
        return this;
      }

      const rect = this.containerEl.getBoundingClientRect();
      const width = Math.max(1, Math.floor(options.width || rect.width || 720));
      const height = Math.max(1, Math.floor(options.height || rect.height || 1280));

      this.app = new window.PIXI.Application({
        width,
        height,
        backgroundAlpha: 0,
        antialias: true,
        autoDensity: true,
        resolution: Math.min(2, window.devicePixelRatio || 1),
      });

      // Put canvas on top of the video (pointer-events none)
      const view = this.app.view;
      view.style.position = 'absolute';
      view.style.inset = '0';
      view.style.width = '100%';
      view.style.height = '100%';
      view.style.pointerEvents = 'none';
      view.style.zIndex = '20';

      // Ensure container is positioned
      const cs = window.getComputedStyle(this.containerEl);
      if (cs.position === 'static') {
        this.containerEl.style.position = 'relative';
      }
      this.containerEl.appendChild(view);

      this.overlayRoot = new window.PIXI.Container();
      this.app.stage.addChild(this.overlayRoot);

      this._buildOverlays();

      // Animation loop
      this.app.ticker.add((delta) => {
        this._tick(delta);
      });

      // Resize observer: keep canvas sized to container
      try {
        const ro = new ResizeObserver(() => this._resizeToContainer());
        ro.observe(this.containerEl);
        this._resizeObserver = ro;
      } catch (_) {}

      this.isInitialized = true;
      // Use warn so it’s visible in demo console logs.
      console.warn('[KellyPixiCompositor] Initialized');
      try { window.__KELLY_PIXI_READY = true; } catch (_) {}
      return this;
    },

    attachVideo(videoEl) {
      this.videoEl = videoEl || null;
      return this;
    },

    setEnabled(enabled) {
      this.isEnabled = !!enabled;
      if (this.app?.view) this.app.view.style.display = this.isEnabled ? '' : 'none';
      return this;
    },

    setFaceAnchor(anchor) {
      if (!anchor) return this;
      this.anchor = {
        x: clamp01(anchor.x ?? this.anchor.x),
        y: clamp01(anchor.y ?? this.anchor.y),
        scale: clamp(anchor.scale ?? this.anchor.scale, 0.5, 2.0),
        rotation: clamp(anchor.rotation ?? this.anchor.rotation, -0.5, 0.5),
      };
      return this;
    },

    setBlendshapes(blendshapes) {
      this.lastBlendshapes = blendshapes || {};
      return this;
    },

    _resizeToContainer() {
      if (!this.app || !this.containerEl) return;
      const rect = this.containerEl.getBoundingClientRect();
      const w = Math.max(1, Math.floor(rect.width));
      const h = Math.max(1, Math.floor(rect.height));
      if (w === this.app.renderer.width && h === this.app.renderer.height) return;
      this.app.renderer.resize(w, h);
    },

    _buildOverlays() {
      // Mouth group (procedural)
      const mouth = new window.PIXI.Container();
      mouth.name = 'mouth';
      this.mouth = mouth;

      // Mouth interior
      const mouthInterior = new window.PIXI.Graphics();
      mouthInterior.name = 'mouthInterior';

      // Lips
      const upperLip = new window.PIXI.Graphics();
      upperLip.name = 'upperLip';
      const lowerLip = new window.PIXI.Graphics();
      lowerLip.name = 'lowerLip';
      this.upperLip = upperLip;
      this.lowerLip = lowerLip;

      // Teeth highlight (subtle)
      const teeth = new window.PIXI.Graphics();
      teeth.name = 'teeth';
      this.teeth = teeth;

      mouth.addChild(mouthInterior);
      mouth.addChild(teeth);
      mouth.addChild(upperLip);
      mouth.addChild(lowerLip);

      this._mouthInterior = mouthInterior;

      // Blink overlays (subtle eyelid sweeps)
      const blinkLeft = new window.PIXI.Graphics();
      const blinkRight = new window.PIXI.Graphics();
      blinkLeft.name = 'blinkLeft';
      blinkRight.name = 'blinkRight';
      this.blinkLeft = blinkLeft;
      this.blinkRight = blinkRight;

      this.overlayRoot.addChild(blinkLeft);
      this.overlayRoot.addChild(blinkRight);
      this.overlayRoot.addChild(mouth);

      // Start hidden until we have anchors and are enabled
      mouth.visible = true;

      // Optional debug marker to prove overlay is rendering (opt-in)
      try {
        const isDebug = (typeof location !== 'undefined') && location.search.includes('pixiDebug=1');
        if (isDebug) {
          const marker = new window.PIXI.Graphics();
          marker.name = 'debugMarker';
          marker.beginFill(0xff3b30, 0.9);
          marker.drawCircle(0, 0, 10);
          marker.endFill();
          this._debugMarker = marker;
          this.overlayRoot.addChild(marker);
        }
      } catch (_) {}
    },

    _tick(delta) {
      if (!this.isEnabled || !this.app) return;

      // Update blink
      this._updateBlink(delta);

      // Update overlay positions/sizes
      this._renderOverlaysFromBlendshapes(this.lastBlendshapes || {});
    },

    _updateBlink(delta) {
      // Simple deterministic blink loop
      // - Every ~4-6 seconds blink once (fast close + open)
      const dt = (delta || 1) / 60;
      this._blinkTimer += dt;

      if (this._blinkState === 0) {
        const next = 4.0 + (Math.sin(this._blinkTimer * 0.7) * 1.0 + 1.0) * 1.0; // ~4-6s
        if (this._blinkTimer > next) {
          this._blinkState = 1;
          this._blinkPhase = 0;
        }
      } else {
        this._blinkPhase += dt;
        if (this._blinkPhase >= 0.16) {
          this._blinkState = 0;
          this._blinkTimer = 0;
        }
      }
    },

    _getBlinkAmount() {
      if (this._blinkState === 0) return 0;
      // ease in/out over 0.16s
      const t = clamp(this._blinkPhase / 0.16, 0, 1);
      const eased = t < 0.5 ? (t * 2) : (2 - t * 2);
      return clamp(eased, 0, 1);
    },

    _renderOverlaysFromBlendshapes(bs) {
      const r = this.app.renderer;
      if (!r) return;

      // Anchor in pixels
      const ax = this.anchor.x * r.width;
      const ay = this.anchor.y * r.height;
      const s = this.anchor.scale;

      // Convert blendshapes (0..100-ish) into mouth params
      const jawOpen = clamp((bs.jawOpen ?? bs.mouthOpen ?? 0) / 100, 0, 1);
      const funnel = clamp((bs.mouthFunnel ?? 0) / 100, 0, 1);
      const pucker = clamp((bs.mouthPucker ?? bs.mouthPucker ?? 0) / 100, 0, 1);
      const stretch = clamp(((bs.mouthStretchLeft ?? 0) + (bs.mouthStretchRight ?? 0)) / 200, 0, 1);

      // Mouth size in pixels (tuned visually)
      const baseW = 150 * s;
      const baseH = 60 * s;
      const openH = baseH + jawOpen * 120 * s;
      const w = baseW + stretch * 80 * s - pucker * 40 * s;
      const h = Math.max(12 * s, openH);

      // Mouth position offsets relative to anchor
      const mx = ax;
      const my = ay + 80 * s; // mouth sits below the anchor center

      // Draw mouth interior
      const mouthInterior = this._mouthInterior;
      mouthInterior.clear();
      mouthInterior.beginFill(0x5b0c0c, 0.88);
      mouthInterior.drawRoundedRect(-w / 2, -h / 2, w, h, 18 * s);
      mouthInterior.endFill();

      // Teeth (only when somewhat open)
      this.teeth.clear();
      if (jawOpen > 0.18) {
        this.teeth.beginFill(0xffffff, 0.22);
        this.teeth.drawRoundedRect(-w * 0.28, -h * 0.35, w * 0.56, h * 0.18, 8 * s);
        this.teeth.endFill();
      }

      // Lips (subtle, skin-ish)
      const lipAlpha = 0.28 + funnel * 0.12;
      const lipColor = 0xa86b6b;
      const upper = this.upperLip;
      upper.clear();
      upper.beginFill(lipColor, lipAlpha);
      upper.drawRoundedRect(-w / 2, -h / 2 - 10 * s, w, 18 * s, 10 * s);
      upper.endFill();

      const lower = this.lowerLip;
      lower.clear();
      lower.beginFill(lipColor, lipAlpha);
      lower.drawRoundedRect(-w / 2, h / 2 - 8 * s, w, 18 * s, 10 * s);
      lower.endFill();

      // Place mouth container
      this.mouth.x = mx;
      this.mouth.y = my;
      this.mouth.rotation = this.anchor.rotation || 0;

      if (this._debugMarker) {
        this._debugMarker.x = ax;
        this._debugMarker.y = ay;
      }

      // Blink overlays: draw two semi-transparent eyelid bands
      const blink = this._getBlinkAmount();
      const blinkOpacity = 0.18 + blink * 0.6;
      const eyelidH = (4 + blink * 40) * s;
      const eyeW = 120 * s;

      // Left/right eye positions relative to anchor
      const eyeY = ay - 10 * s;
      const eyeDX = 110 * s;

      this.blinkLeft.clear();
      this.blinkRight.clear();
      if (blink > 0.01) {
        this.blinkLeft.beginFill(0xffffff, blinkOpacity);
        this.blinkLeft.drawRoundedRect(ax - eyeDX - eyeW / 2, eyeY - eyelidH / 2, eyeW, eyelidH, 10 * s);
        this.blinkLeft.endFill();

        this.blinkRight.beginFill(0xffffff, blinkOpacity);
        this.blinkRight.drawRoundedRect(ax + eyeDX - eyeW / 2, eyeY - eyelidH / 2, eyeW, eyelidH, 10 * s);
        this.blinkRight.endFill();
      }
    },
  };

  window.KellyPixiCompositor = KellyPixiCompositor;
})();


