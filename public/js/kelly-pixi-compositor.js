/**
 * Kelly Pixi Compositor (WebGL overlay layer)
 * - Renders a procedural mouth + blink overlay on top of the HeyGen full-frame video (white background).
 * - Uses static face anchors (demo-safe). No segmentation required.
 *
 * Requirements:
 * - pixi.js v7 OR v8 available as global `PIXI` (loaded via CDN in learn.html)
 *
 * API:
 * - KellyPixiCompositor.init({ containerEl, width, height }) - returns Promise in v8
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
    // Calibrated for `public/kelly/videos/001/welcome.mp4` (white background talking head)
    // Based on screenshot analysis 2025-12-21:
    // - Eyes at ~38% from top
    // - Face center at ~42%
    // - Mouth at ~56%
    // Anchor is set to face center; mouth/eye offsets are relative to this.
    x: 0.5,     // Horizontal center (Kelly is centered)
    y: 0.42,    // Vertical face center (between eyes and nose)
    scale: 0.8, // Slightly smaller overlays for subtlety
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
    _initPromise: null,

    /**
     * Initialize the compositor. Works with both PixiJS v7 (sync) and v8 (async).
     * Always returns a Promise for consistent API.
     */
    init(options = {}) {
      console.log('[KellyPixiCompositor] init() called, options:', options);
      
      if (this.isInitialized) {
        console.log('[KellyPixiCompositor] Already initialized, returning');
        return Promise.resolve(this);
      }
      if (this._initPromise) {
        console.log('[KellyPixiCompositor] Init already in progress, returning promise');
        return this._initPromise;
      }

      if (typeof window === 'undefined' || !window.PIXI) {
        console.warn('[KellyPixiCompositor] PIXI not found; compositor disabled. window.PIXI =', window.PIXI);
        return Promise.resolve(this);
      }
      
      console.log('[KellyPixiCompositor] PIXI found, version info:', window.PIXI.VERSION || 'unknown');

      this.containerEl =
        options.containerEl ||
        (typeof document !== 'undefined' ? document.getElementById('kelly-stage') : null);

      if (!this.containerEl) {
        console.warn('[KellyPixiCompositor] container not found');
        return Promise.resolve(this);
      }

      const rect = this.containerEl.getBoundingClientRect();
      const width = Math.max(1, Math.floor(options.width || rect.width || 720));
      const height = Math.max(1, Math.floor(options.height || rect.height || 1280));

      // PixiJS v8 uses async init; v7 uses constructor options
      const pixiOptions = {
        width,
        height,
        backgroundAlpha: 0,
        antialias: true,
        autoDensity: true,
        resolution: Math.min(2, window.devicePixelRatio || 1),
      };

      this._initPromise = this._createApp(pixiOptions).then(() => {
        // Put canvas on top of the video (pointer-events none)
        const view = this.app.canvas || this.app.view; // v8 uses .canvas, v7 uses .view
        if (!view) {
          console.warn('[KellyPixiCompositor] No canvas/view found on app');
          return this;
        }
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
        // Use warn so it's visible in demo console logs.
        console.warn('[KellyPixiCompositor] Initialized');
        try { window.__KELLY_PIXI_READY = true; } catch (_) {}
        return this;
      }).catch((err) => {
        console.error('[KellyPixiCompositor] Init failed:', err);
        return this;
      });

      return this._initPromise;
    },

    /**
     * Create PIXI.Application - handles both v7 (sync) and v8 (async) APIs.
     */
    async _createApp(options) {
      const PIXI = window.PIXI;
      console.log('[KellyPixiCompositor] _createApp called with options:', options);
      
      // Check if this is PixiJS v8 (has async init method)
      const app = new PIXI.Application();
      console.log('[KellyPixiCompositor] Created Application instance, has init():', typeof app.init === 'function');
      
      if (typeof app.init === 'function') {
        // PixiJS v8: async init
        console.log('[KellyPixiCompositor] Using PixiJS v8 async init...');
        await app.init(options);
        console.log('[KellyPixiCompositor] PixiJS v8 init complete');
        this.app = app;
      } else {
        // PixiJS v7: constructor takes options directly
        // Need to recreate with options
        console.log('[KellyPixiCompositor] Using PixiJS v7 sync init...');
        this.app = new PIXI.Application(options);
        console.log('[KellyPixiCompositor] PixiJS v7 init complete');
      }
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

      // Mouth size in pixels (calibrated for subtlety - blend with video)
      // Kelly's actual mouth is approximately 80-100px wide in the video
      const baseW = 90 * s;
      const baseH = 25 * s;
      const openH = baseH + jawOpen * 50 * s;
      const w = baseW + stretch * 30 * s - pucker * 20 * s;
      const h = Math.max(6 * s, openH);

      // Mouth position offsets relative to anchor
      // Calibrated: mouth is ~14% below face center in Kelly's video
      // For 1000px viewport: anchor at 42% = 420px, mouth at 56% = 560px
      // Offset = 140px at scale 1.0, so 175px base * scale
      const mx = ax;
      const my = ay + 175 * s; // mouth sits below the anchor center

      // Draw mouth interior (very subtle - blend with video)
      const mouthInterior = this._mouthInterior;
      mouthInterior.clear();
      mouthInterior.beginFill(0x3d1515, 0.35); // Darker, lower opacity
      mouthInterior.drawRoundedRect(-w / 2, -h / 2, w, h, 8 * s);
      mouthInterior.endFill();

      // Teeth hint (only when mouth is quite open)
      this.teeth.clear();
      if (jawOpen > 0.35) {
        this.teeth.beginFill(0xf5f0f0, 0.12); // Very subtle teeth
        this.teeth.drawRoundedRect(-w * 0.25, -h * 0.3, w * 0.5, h * 0.15, 4 * s);
        this.teeth.endFill();
      }

      // Lips (very subtle highlight - should barely be noticeable)
      const lipAlpha = 0.08 + funnel * 0.06; // Much lower opacity
      const lipColor = 0xc99a9a; // Lighter, more natural lip color
      const upper = this.upperLip;
      upper.clear();
      upper.beginFill(lipColor, lipAlpha);
      upper.drawRoundedRect(-w / 2, -h / 2 - 4 * s, w, 8 * s, 4 * s);
      upper.endFill();

      const lower = this.lowerLip;
      lower.clear();
      lower.beginFill(lipColor, lipAlpha);
      lower.drawRoundedRect(-w / 2, h / 2 - 3 * s, w, 8 * s, 4 * s);
      lower.endFill();

      // Place mouth container
      this.mouth.x = mx;
      this.mouth.y = my;
      this.mouth.rotation = this.anchor.rotation || 0;

      if (this._debugMarker) {
        this._debugMarker.x = ax;
        this._debugMarker.y = ay;
      }

      // Blink overlays: very subtle eyelid bands (should barely be visible)
      const blink = this._getBlinkAmount();
      const blinkOpacity = 0.05 + blink * 0.25; // Much more subtle
      // Eye overlay size (calibrated for subtlety)
      const eyelidH = (2 + blink * 20) * s;
      const eyeW = 60 * s;

      // Left/right eye positions relative to anchor
      // Calibrated: eyes are ~4% above face center in Kelly's video
      // For 1000px viewport: anchor at 42% = 420px, eyes at 38% = 380px
      // Offset = -40px at scale 1.0, so -50px base * scale
      const eyeY = ay - 50 * s;
      const eyeDX = 85 * s; // Eye horizontal spacing (reduced for realistic proportion)

      this.blinkLeft.clear();
      this.blinkRight.clear();
      if (blink > 0.02) {
        // Use skin tone for eyelids (Kelly's skin tone)
        const skinColor = 0xe8d4c4;
        this.blinkLeft.beginFill(skinColor, blinkOpacity);
        this.blinkLeft.drawRoundedRect(ax - eyeDX - eyeW / 2, eyeY - eyelidH / 2, eyeW, eyelidH, 4 * s);
        this.blinkLeft.endFill();

        this.blinkRight.beginFill(skinColor, blinkOpacity);
        this.blinkRight.drawRoundedRect(ax + eyeDX - eyeW / 2, eyeY - eyelidH / 2, eyeW, eyelidH, 4 * s);
        this.blinkRight.endFill();
      }
    },
  };

  window.KellyPixiCompositor = KellyPixiCompositor;
})();
