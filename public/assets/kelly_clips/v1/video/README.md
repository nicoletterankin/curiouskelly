# Delivery-Ready Kelly Clips

Transcode each master render from `renders/kelly_clips_v1/video/` into an H.264/H.265 MP4 (1080p60) and store it here using the clip ID:

```
clip01_warm_welcome.mp4
clip02_awe_and_wonder.mp4
...
```

After copying the files:
1. Run `shasum -a 256 <file>` (or `CertUtil -hashfile` on Windows) and record the hash + duration in `assets/kelly_clips/v1/render_log.json`.
2. Update `assets/kelly_clips/v1/manifest.template.json` with the new metadata so the lesson player can reference the clip.
3. Keep the MP4 bitrate ≥20 Mbps to preserve facial detail; we can generate lighter proxies later if needed.








