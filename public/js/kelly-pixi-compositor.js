/**
 * Kelly Pixi Compositor (WebGL overlay layer)
 * - Renders a procedural mouth + blink overlay on top of Kelly (video OR static image).
 * - Uses static face anchors (demo-safe). No segmentation required.
 * - Supports "talking photo" mode with static head images.
 *
 * Requirements:
 * - pixi.js v7 OR v8 available as global `PIXI` (loaded via CDN in learn.html)
 *
 * API:
 * - KellyPixiCompositor.init({ containerEl, width, height }) - returns Promise in v8
 * - KellyPixiCompositor.attachVideo(videoEl) - for video base layer
 * - KellyPixiCompositor.attachImage(imgPath, archetype) - for static image base layer
 * - KellyPixiCompositor.setFaceAnchor({ x, y, scale, rotation })
 * - KellyPixiCompositor.setBlendshapes(blendshapes)
 * - KellyPixiCompositor.setEnabled(true/false)
 *
 * Modes:
 * - VIDEO mode: overlays on top of playing video element
 * - IMAGE mode: overlays on top of static Kelly head image (talking photo)
 *
 * Debug:
 * - Add `?pixiDebug=1` to render a red anchor dot so you can visually confirm overlays are rendering.
 */
(() => {
  // ALWAYS log script load (critical for debugging)
  console.log('[Pixi] 🎭 kelly-pixi-compositor.js LOADED, v=20251222m - production hardened');
  
  const DEBUG =
    (typeof window !== 'undefined' && !!window.__KELLY_PIXI_DEBUG) ||
    (typeof location !== 'undefined' && (location.search.includes('pixiDebug=1') || location.search.includes('hybridDebug=1')));

  console.log('[Pixi] DEBUG mode:', DEBUG);
  
  const dlog = (...args) => { if (DEBUG) console.log(...args); };
  const dwarn = (...args) => { if (DEBUG) console.warn(...args); };
  
  // Face anchor presets for different Kelly assets
  // All 1024x1024 head images use the same calibration (consistent face position)
  const ANCHOR_PRESETS = {
    // For 1024x1024 static head images (all archetypes)
    // Calibrated from kelly_explorer_head.png live test 2025-12-22:
    // VISUAL OBSERVATION:
    // - Nose (where red dot was): ~36% from top
    // - Actual mouth/smile: ~56% from top
    // - Chin (where overlay was): ~60% from top
    // SOLUTION: Move anchor to 40%, mouth offset to 16% = 56% total
    head_image: {
      x: 0.5,      // Horizontal center
      y: 0.40,     // Anchor at mid-face (between nose and mouth)
      scale: 1.0,  // Full size for 1024x1024
      rotation: 0,
      mouthOffsetY: 0.16,  // Mouth at 56% from top (40% + 16%)
      eyeOffsetY: -0.02,   // Eyes slightly above anchor
    },
    // For HeyGen welcome.mp4 video
    video_heygen: {
      x: 0.5,
      y: 0.42,
      scale: 0.8,
      rotation: 0,
      mouthOffsetY: 0.14,
      eyeOffsetY: -0.04,
    },
  };
  
  const DEFAULT_ANCHOR = { ...ANCHOR_PRESETS.head_image };

  // Overlay opacity presets (tuned for subtle blending with Kelly's face)
  // Values are 0.0 (transparent) to 1.0 (opaque)
  // These can be adjusted based on visual testing with real learners
  const OPACITY_PRESETS = {
    mouthInterior: 0.35,      // Mouth opening (35% - visible but subtle)
    teeth: 0.12,               // Teeth hint when mouth open (12% - very subtle)
    lipBase: 0.08,             // Base lip opacity (8% - barely visible)
    lipMax: 0.14,              // Max lip opacity when funneled (14%)
    blinkMin: 0.05,            // Minimum blink opacity (5% - very subtle)
    blinkMax: 0.30,            // Maximum blink opacity (30% - noticeable but natural)
    debugMarker: 0.9,          // Debug red dot (90% - highly visible for calibration)
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
    mode: 'none',       // 'none' | 'video' | 'image'
    containerEl: null,
    videoEl: null,
    imageEl: null,      // For static image mode
    imageSprite: null,  // PixiJS sprite for image
    archetype: null,    // Current archetype (for head image selection)
    app: null,
    overlayRoot: null,
    mouth: null,
    upperLip: null,
    lowerLip: null,
    teeth: null,
    blinkLeft: null,
    blinkRight: null,
    eyebrowLeft: null,
    eyebrowRight: null,
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
      dlog('[KellyPixiCompositor] init() called, options:', options);
      
      if (this.isInitialized) {
        dlog('[KellyPixiCompositor] Already initialized, returning');
        return Promise.resolve(this);
      }
      if (this._initPromise) {
        dlog('[KellyPixiCompositor] Init already in progress, returning promise');
        return this._initPromise;
      }

      if (typeof window === 'undefined' || !window.PIXI) {
        dwarn('[KellyPixiCompositor] PIXI not found; compositor disabled. window.PIXI =', window.PIXI);
        return Promise.resolve(this);
      }
      
      dlog('[KellyPixiCompositor] PIXI found, version info:', window.PIXI.VERSION || 'unknown');

      this.containerEl =
        options.containerEl ||
        (typeof document !== 'undefined' ? document.getElementById('kelly-stage') : null);

      if (!this.containerEl) {
        dwarn('[KellyPixiCompositor] container not found');
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

      this._initPromise = this._createApp(pixiOptions).then((canvas) => {
        console.log('[Pixi] _createApp resolved, canvas:', canvas, 'this.app:', this.app);
        
        if (!this.app) {
          console.error('[Pixi] FATAL: this.app is undefined after _createApp!');
          return this;
        }
        
        if (!canvas) {
          console.error('[Pixi] FATAL: No canvas returned from _createApp!');
          return this;
        }
        
        // Style the canvas for overlay positioning
        canvas.style.position = 'absolute';
        canvas.style.inset = '0';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.pointerEvents = 'none';
        canvas.style.zIndex = '20';

        // Ensure container is positioned
        const cs = window.getComputedStyle(this.containerEl);
        if (cs.position === 'static') {
          this.containerEl.style.position = 'relative';
        }
        this.containerEl.appendChild(canvas);
        console.log('[Pixi] Canvas appended to container');

        this.overlayRoot = new window.PIXI.Container();
        this.app.stage.addChild(this.overlayRoot);

        this._buildOverlays();
        console.log('[Pixi] Overlays built');

        // Animation loop (throttled for performance)
        // Only update when compositor is enabled and visible
        this.app.ticker.add((delta) => {
          if (this.isEnabled && this.isInitialized) {
            this._tick(delta);
          }
        });
        
        // Performance: Stop rendering when tab is hidden
        if (typeof document !== 'undefined') {
          document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
              this.app.ticker.stop();
            } else {
              this.app.ticker.start();
            }
          });
        }

        // Resize observer: keep canvas sized to container
        try {
          const ro = new ResizeObserver(() => this._resizeToContainer());
          ro.observe(this.containerEl);
          this._resizeObserver = ro;
        } catch (_) {}

        this.isInitialized = true;
        console.log('[Pixi] ✅ Compositor READY - Kelly\'s mouth can now move!');
        try { window.__KELLY_PIXI_READY = true; } catch (_) {}
        return this;
      }).catch((err) => {
        console.error('[Pixi] Compositor init FAILED:', err);
        return this;
      });

      return this._initPromise;
    },

    /**
     * Create PIXI.Application - handles both v7 (sync) and v8 (async) APIs.
     * Returns the canvas element on success.
     */
    async _createApp(options = {}) {
      const PIXI = window.PIXI;
      
      // Always log init attempt (critical for debugging)
      console.log('[Pixi] _createApp called, PIXI version:', PIXI.VERSION || 'unknown');
      console.log('[Pixi] Options:', JSON.stringify(options));
      
      try {
        // PixiJS v8 requires async init
        const app = new PIXI.Application();
        
        if (typeof app.init === 'function') {
          // PixiJS v8 path
          console.log('[Pixi] Using PixiJS v8 async init...');
          
          const initOptions = {
            width: options.width || 1920,
            height: options.height || 1080,
            backgroundAlpha: 0,
            antialias: true,
            autoDensity: true,
            resolution: Math.min(2, window.devicePixelRatio || 1),
            ...options
          };
          
          await app.init(initOptions);
          
          // CRITICAL: Set this.app AFTER successful init
          this.app = app;
          
          console.log('[Pixi] v8 init SUCCESS');
          console.log('[Pixi] Canvas:', this.app.canvas);
          
          if (!this.app.canvas) {
            throw new Error('PixiJS v8 init succeeded but canvas is null');
          }
          
          return this.app.canvas;
          
        } else {
          // PixiJS v7 path (sync constructor)
          console.log('[Pixi] Using PixiJS v7 sync init...');
          
          this.app = new PIXI.Application({
            width: options.width || 1920,
            height: options.height || 1080,
            backgroundAlpha: 0,
            antialias: true,
            autoDensity: true,
            resolution: Math.min(2, window.devicePixelRatio || 1),
            ...options
          });
          
          console.log('[Pixi] v7 init SUCCESS');
          console.log('[Pixi] View:', this.app.view);
          
          return this.app.view;
        }
        
      } catch (error) {
        console.error('[Pixi] Primary init FAILED:', error);
        
        // Fallback: try legacy sync pattern for older Pixi versions
        try {
          console.log('[Pixi] Attempting legacy sync fallback...');
          this.app = new PIXI.Application({
            width: options.width || 1920,
            height: options.height || 1080,
            backgroundAlpha: 0,
            antialias: true,
            ...options
          });
          console.log('[Pixi] Legacy fallback SUCCESS');
          return this.app.view || this.app.canvas;
        } catch (legacyError) {
          console.error('[Pixi] Legacy fallback also FAILED:', legacyError);
          throw legacyError;
        }
      }
    },

    attachVideo(videoEl) {
      this.videoEl = videoEl || null;
      this.mode = videoEl ? 'video' : 'none';
      this.anchor = { ...ANCHOR_PRESETS.video_heygen };
      console.log('[Pixi] Attached video, mode:', this.mode);
      return this;
    },

    /**
     * Attach a static Kelly head image (talking photo mode).
     * @param {string} archetype - e.g. 'storyteller', 'scientist', 'explorer'
     * @returns {Promise<this>}
     */
    async attachImage(archetype = 'default') {
      const imagePath = `/kelly/heads/kelly_${archetype}_head.png`;
      console.log('[Pixi] Attaching image:', imagePath);
      
      try {
        // Create image element
        const img = new Image();
        img.crossOrigin = 'anonymous';
        
        await new Promise((resolve, reject) => {
          img.onload = resolve;
          img.onerror = reject;
          img.src = imagePath;
        });
        
        this.imageEl = img;
        this.archetype = archetype;
        this.mode = 'image';
        this.anchor = { ...ANCHOR_PRESETS.head_image };
        
        // If PixiJS is initialized, add the image as a sprite
        if (this.app && this.app.stage) {
          // Remove existing image sprite
          if (this.imageSprite) {
            this.app.stage.removeChild(this.imageSprite);
          }
          
          // Create texture and sprite
          const texture = window.PIXI.Texture.from(img);
          this.imageSprite = new window.PIXI.Sprite(texture);
          
          // Scale to fit container
          const scale = Math.min(
            this.app.renderer.width / img.width,
            this.app.renderer.height / img.height
          );
          this.imageSprite.scale.set(scale);
          
          // Center the image
          this.imageSprite.x = (this.app.renderer.width - img.width * scale) / 2;
          this.imageSprite.y = (this.app.renderer.height - img.height * scale) / 2;
          
          // Add as first child (behind overlays)
          this.app.stage.addChildAt(this.imageSprite, 0);
          
          console.log('[Pixi] Image sprite added, scale:', scale);
        }
        
        console.log('[Pixi] Attached image, mode:', this.mode, 'archetype:', archetype);
        return this;
        
      } catch (err) {
        console.error('[Pixi] Failed to attach image:', err);
        return this;
      }
    },

    setEnabled(enabled) {
      this.isEnabled = !!enabled;
      const canvas = this.app?.canvas || this.app?.view;
      if (canvas) canvas.style.display = this.isEnabled ? '' : 'none';
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
      try {
        this.lastBlendshapes = blendshapes || {};
      } catch (e) {
        console.warn('[Pixi] setBlendshapes error:', e);
        this.lastBlendshapes = {};
      }
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

      // Eyebrow overlays (for expression system)
      const eyebrowLeft = new window.PIXI.Graphics();
      const eyebrowRight = new window.PIXI.Graphics();
      eyebrowLeft.name = 'eyebrowLeft';
      eyebrowRight.name = 'eyebrowRight';
      this.eyebrowLeft = eyebrowLeft;
      this.eyebrowRight = eyebrowRight;

      this.overlayRoot.addChild(blinkLeft);
      this.overlayRoot.addChild(blinkRight);
      this.overlayRoot.addChild(eyebrowLeft);
      this.overlayRoot.addChild(eyebrowRight);
      this.overlayRoot.addChild(mouth);

      // Optional debug marker to prove overlay is rendering (opt-in)
      try {
        const isDebug = (typeof location !== 'undefined') && location.search.includes('pixiDebug=1');
        if (isDebug) {
          const marker = new window.PIXI.Graphics();
          marker.name = 'debugMarker';
          marker.beginFill(0xff3b30, OPACITY_PRESETS.debugMarker);
          marker.drawCircle(0, 0, 10);
          marker.endFill();
          this._debugMarker = marker;
          this.overlayRoot.addChild(marker);
        }
      } catch (_) {}
    },

    _tick(delta) {
      if (!this.isEnabled || !this.app) return;
      
      // Performance: Skip rendering if blendshapes haven't changed significantly
      // This reduces unnecessary redraws when audio is silent
      const hasBlendshapes = Object.keys(this.lastBlendshapes || {}).length > 0;
      if (!hasBlendshapes && this._blinkState === 0) {
        // Only update blink timer, skip rendering
        this._updateBlink(delta);
        return;
      }
      
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
      try {
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

      // Mouth position - ABSOLUTE percentage of viewport height
      // Based on visual analysis of kelly_explorer_head.png:
      // Kelly's mouth/smile is at 56% from top of frame
      // Using ABSOLUTE position to avoid anchor offset confusion
      const MOUTH_Y_ABSOLUTE = 0.56;  // 56% from top = Kelly's mouth
      const mx = ax;  // horizontal center (from anchor)
      const my = MOUTH_Y_ABSOLUTE * r.height;  // ABSOLUTE vertical position

      // Enhanced mouth shape: more natural oval/ellipse instead of rectangle
      // Corner radius varies with mouth state (more rounded when closed, sharper when open)
      const cornerRadius = (8 + (1 - jawOpen) * 4) * s; // 8-12px radius
      const mouthShape = jawOpen > 0.1 ? 'ellipse' : 'roundedRect'; // Ellipse when open, rounded rect when closed

      // Draw mouth interior (very subtle - blend with video)
      const mouthInterior = this._mouthInterior;
      mouthInterior.clear();
      mouthInterior.beginFill(0x3d1515, OPACITY_PRESETS.mouthInterior);
      if (mouthShape === 'ellipse') {
        // More natural oval shape when mouth is open
        mouthInterior.drawEllipse(0, 0, w / 2, h / 2);
      } else {
        // Rounded rectangle when mouth is closed
        mouthInterior.drawRoundedRect(-w / 2, -h / 2, w, h, cornerRadius);
      }
      mouthInterior.endFill();

      // Teeth hint (only when mouth is quite open)
      this.teeth.clear();
      if (jawOpen > 0.35) {
        this.teeth.beginFill(0xf5f0f0, OPACITY_PRESETS.teeth);
        this.teeth.drawRoundedRect(-w * 0.25, -h * 0.3, w * 0.5, h * 0.15, 4 * s);
        this.teeth.endFill();
      }

      // Lips (very subtle highlight - should barely be noticeable)
      // Enhanced: curved lip shapes with better separation
      const lipAlpha = OPACITY_PRESETS.lipBase + funnel * (OPACITY_PRESETS.lipMax - OPACITY_PRESETS.lipBase);
      const lipColor = 0xc99a9a; // Lighter, more natural lip color
      const lipThickness = (4 + jawOpen * 2) * s; // Thicker when mouth opens
      const lipCurve = 2 * s; // Subtle curve for natural lip shape
      
      const upper = this.upperLip;
      upper.clear();
      upper.beginFill(lipColor, lipAlpha);
      // Upper lip: curved shape with slight dip in center (Cupid's bow effect)
      upper.moveTo(-w / 2, -h / 2);
      upper.quadraticCurveTo(-w / 4, -h / 2 - lipCurve, 0, -h / 2); // Left curve
      upper.quadraticCurveTo(w / 4, -h / 2 - lipCurve, w / 2, -h / 2); // Right curve
      upper.lineTo(w / 2, -h / 2 + lipThickness);
      upper.quadraticCurveTo(w / 4, -h / 2 + lipThickness + lipCurve, 0, -h / 2 + lipThickness);
      upper.quadraticCurveTo(-w / 4, -h / 2 + lipThickness + lipCurve, -w / 2, -h / 2 + lipThickness);
      upper.closePath();
      upper.endFill();

      const lower = this.lowerLip;
      lower.clear();
      lower.beginFill(lipColor, lipAlpha);
      // Lower lip: fuller, more rounded shape
      lower.drawRoundedRect(-w / 2, h / 2 - lipThickness, w, lipThickness, 4 * s);
      lower.endFill();

      // Place mouth container
      this.mouth.x = mx;
      this.mouth.y = my;
      this.mouth.rotation = this.anchor.rotation || 0;

      // Debug marker shows MOUTH position (not anchor) for visual verification
      if (this._debugMarker) {
        this._debugMarker.x = mx;  // Mouth x (horizontal center)
        this._debugMarker.y = my;  // Mouth y (56% absolute - Kelly's actual mouth)
      }

      // Blink overlays: very subtle eyelid bands (should barely be visible)
      const blink = this._getBlinkAmount();
      const blinkOpacity = OPACITY_PRESETS.blinkMin + blink * (OPACITY_PRESETS.blinkMax - OPACITY_PRESETS.blinkMin);
      // Eye overlay size (calibrated for subtlety)
      const eyelidH = (2 + blink * 20) * s;
      const eyeW = 60 * s;

      // Left/right eye positions relative to anchor
      // Uses eyeOffsetY from anchor preset (percentage of viewport height)
      const eyeOffsetY = this.anchor.eyeOffsetY || -0.04;
      const eyeY = ay + (eyeOffsetY * r.height); // eyes offset from anchor
      const eyeDX = 85 * s; // Eye horizontal spacing

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

      // Eyebrow overlays (for expression system)
      // Get eyebrow raise from blendshapes (0-100 range, normalized to 0-1)
      const eyebrowRaise = clamp(
        Math.max(
          (bs.browInnerUp ?? 0) / 100,
          (bs.browOuterUpLeft ?? 0) / 100,
          (bs.browOuterUpRight ?? 0) / 100
        ),
        0,
        1
      );
      if (eyebrowRaise > 0.01 && this.eyebrowLeft && this.eyebrowRight) {
        const eyebrowColor = 0x8b6f5e; // Brown eyebrow color
        const eyebrowW = 50 * s;
        const eyebrowH = 4 * s;
        const eyebrowY = eyeY - 35 * s; // Above eyes
        const eyebrowRaiseAmount = eyebrowRaise * 8 * s; // Raise up to 8px when raised
        const eyebrowOpacity = 0.4 + eyebrowRaise * 0.3; // 40-70% opacity

        this.eyebrowLeft.clear();
        this.eyebrowLeft.beginFill(eyebrowColor, eyebrowOpacity);
        this.eyebrowLeft.drawRoundedRect(
          ax - eyeDX - eyebrowW / 2,
          eyebrowY - eyebrowRaiseAmount - eyebrowH / 2,
          eyebrowW,
          eyebrowH,
          2 * s
        );
        this.eyebrowLeft.endFill();

        this.eyebrowRight.clear();
        this.eyebrowRight.beginFill(eyebrowColor, eyebrowOpacity);
        this.eyebrowRight.drawRoundedRect(
          ax + eyeDX - eyebrowW / 2,
          eyebrowY - eyebrowRaiseAmount - eyebrowH / 2,
          eyebrowW,
          eyebrowH,
          2 * s
        );
        this.eyebrowRight.endFill();
      } else if (this.eyebrowLeft && this.eyebrowRight) {
        this.eyebrowLeft.clear();
        this.eyebrowRight.clear();
      }
      } catch (e) {
        console.warn('[Pixi] Render error (non-fatal):', e);
        // Continue rendering - don't break the entire system
      }
    },

    /**
     * Set expression (for phase-to-expression mapping)
     * @param {string} expressionName - e.g. 'curious', 'engaged', 'warm'
     */
    setExpression(expressionName) {
      // Expression system will be handled by KellyExpressionBridge
      // This method is a placeholder for future direct expression control
      if (window.KellyExpressionBridge && window.KellyExpressionBridge.isInitialized) {
        window.KellyExpressionBridge.setExpression(expressionName);
      }
      return this;
    },
  };

  window.KellyPixiCompositor = KellyPixiCompositor;
})();
