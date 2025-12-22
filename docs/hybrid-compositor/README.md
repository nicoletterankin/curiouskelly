# Hybrid Compositor Documentation

This folder contains documentation for the **Kelly Hybrid Compositor** - a browser-based system that combines pre-rendered avatar video with real-time TTS audio and WebGL overlays.

## Documents

| File | Purpose |
|------|---------|
| [HYBRID_COMPOSITOR_DIRECTIVE.md](./HYBRID_COMPOSITOR_DIRECTIVE.md) | Main directive, architecture, testing, and next steps |

## Quick Start

```bash
# Test the hybrid demo
open "https://curiouskelly.com/learn.html?hybrid=1&day=1"

# Test with debug overlays
open "https://curiouskelly.com/learn.html?hybrid=1&pixiDebug=1&debug=1&day=1"
```

## Related Files

- `public/js/kelly-pixi-compositor.js` - WebGL overlay renderer
- `public/js/kelly-lipsync.js` - Audio → blendshape analysis
- `infrastructure/cloudflare/tts-worker/` - TTS Cloudflare Worker

## Why This Exists

Apple Education requires **photorealistic, zero-friction, browser-first** avatar delivery. This hybrid system achieves that by:

1. Using HeyGen for the hard parts (photorealistic rendering)
2. Using ElevenLabs for dynamic voice (not pre-recorded)
3. Using PixiJS for real-time mouth/eye animation
4. Deploying via Cloudflare for edge performance

See the directive for full details.

