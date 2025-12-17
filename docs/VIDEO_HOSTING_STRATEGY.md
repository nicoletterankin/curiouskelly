# Video Hosting Strategy

## Problem

HeyGen renders videos and provides temporary CDN URLs that **expire**. We cannot depend on these for production.

Example URL with expiration:
```
https://files2.heygen.ai/.../video.mp4?Expires=1766598156&Signature=...
```

## Solution: Cloudflare R2

### Why R2?

| Feature | R2 | S3 | Vercel Blob |
|---------|----|----|-------------|
| Egress fees | **Free** | $0.09/GB | $0.15/GB |
| Storage | $0.015/GB/mo | $0.023/GB/mo | $0.02/GB/mo |
| CDN | Included | Need CloudFront | Included |
| Custom domain | Yes | Yes | Limited |
| S3 compatible | Yes | Yes | No |

### Architecture

```
[HeyGen renders video]
         ↓
[Download to local]
         ↓
[Upload to Cloudflare R2]
         ↓
[Permanent URL: videos.curiouskelly.com/summary/day-351.mp4]
         ↓
[Use in emails, watch pages, etc.]
```

---

## Setup Steps

### 1. Create R2 Bucket in Cloudflare

1. Go to Cloudflare Dashboard → R2
2. Create bucket: `curious-kelly-videos`
3. Enable public access
4. Add custom domain: `videos.curiouskelly.com`

### 2. Get API Credentials

1. Create R2 API token with read/write permissions
2. Save credentials:
   - `R2_ACCOUNT_ID`
   - `R2_ACCESS_KEY_ID`
   - `R2_SECRET_ACCESS_KEY`
   - `R2_BUCKET_NAME`

### 3. Upload Script

Use the `scripts/upload-video-to-r2.ts` script (created below).

---

## URL Structure

```
videos.curiouskelly.com/
├── summary/
│   ├── day-001.mp4
│   ├── day-001-thumb.jpg
│   ├── day-351.mp4
│   └── day-351-thumb.jpg
├── full/
│   ├── day-351-scientist.mp4
│   ├── day-351-explorer.mp4
│   └── ...
└── manifests/
    └── video-manifest.json
```

---

## Video Manifest

Track all videos and their locations:

```json
{
  "version": "1.0.0",
  "updated": "2025-12-17T18:00:00Z",
  "baseUrl": "https://videos.curiouskelly.com",
  "videos": {
    "day-351": {
      "summary": {
        "mp4": "/summary/day-351.mp4",
        "thumb": "/summary/day-351-thumb.jpg",
        "duration": 107.584,
        "size": 42459535,
        "uploaded": "2025-12-17T18:00:00Z"
      },
      "full": {
        "scientist": "/full/day-351-scientist.mp4",
        "explorer": "/full/day-351-explorer.mp4"
      }
    }
  }
}
```

---

## Immediate Actions (Dec 17)

1. [x] Download all Day 351 videos locally (DONE - in video-backups/day-351/)
2. [x] Set up Cloudflare R2 bucket (`curious-kelly-backups`)
3. [x] Upload videos to R2
4. [x] Update watch page and emails to use R2 URLs
5. [x] Create backup of HeyGen video URLs before they expire

### Current URLs

**R2 Public URL:** `https://pub-29446fb0037f47e49993ebd6b4ed714e.r2.dev`

**Custom Domain (pending):** `videos.curiouskelly.com`

**Example video:** `https://pub-29446fb0037f47e49993ebd6b4ed714e.r2.dev/videos/summary/day-351.mp4`

---

## Cost Estimate

| Item | Cost |
|------|------|
| 365 summary videos (~40MB each) | ~14GB storage = $0.21/month |
| 365 × 12 archetype videos (~40MB each) | ~175GB storage = $2.63/month |
| Bandwidth | **Free** |
| **Total** | **~$3/month** |

---

## Fallback Plan

If Cloudflare R2 isn't available immediately:

1. **GitHub LFS** — Already using, but Vercel doesn't serve LFS files
2. **YouTube Unlisted** — Free, reliable, but loses control
3. **Direct HeyGen backup** — Download all videos before expiry

---

## Priority: CRITICAL

HeyGen URLs expire. Download and backup ALL videos ASAP.

The expiration for Day 351 videos is: `1766598156` (Unix timestamp)
= **December 24, 2025** (about 7 days from now)

---

*Created: December 17, 2025*
