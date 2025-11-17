# HeyGen API Integration Guide

## Overview

HeyGen API allows generating videos from audio files using Avatar IV. This integration is mindful of the 5 minutes/month quota limit.

## Quota Management

- **Limit**: 5 minutes of video per month
- **Tracking**: Quota usage tracked in `lessons/.heygen_quota.json`
- **Reset**: Manual reset required (monthly)

## API Setup

### 1. Get API Credentials

1. Sign up at https://heygen.com
2. Navigate to API settings
3. Generate API key
4. Get Avatar IV ID for Kelly

### 2. Set Environment Variables

```bash
export HEYGEN_API_KEY="your_api_key_here"
export HEYGEN_AVATAR_ID="your_avatar_id_here"
```

Or create `.env` file:
```
HEYGEN_API_KEY=your_api_key_here
HEYGEN_AVATAR_ID=your_avatar_id_here
```

## Usage

### Generate Videos for a Lesson

```bash
python scripts/generate_heygen_videos.py --lesson the-sun --priority-phases welcome mainContent --max-videos 2
```

### Check Video Status

```bash
python scripts/generate_heygen_videos.py --check-status VIDEO_ID
```

### Download Completed Video

```bash
python scripts/generate_heygen_videos.py --download VIDEO_ID --download-to output/video.mp4
```

## Video Generation Strategy

Given the 5-minute quota limit:

1. **Prioritize High-Value Content**
   - Generate videos for welcome and mainContent phases first
   - Skip wisdomMoment initially (can add later)

2. **Selective Generation**
   - Generate for one age variant per lesson (e.g., 18-35)
   - Generate for English only initially
   - Total: ~9 videos (one per lesson) = ~5-10 minutes

3. **Use for Training**
   - Generated videos can be used to train Kelly's animation in iClone
   - Focus on getting lip-sync and expression patterns right

## API Endpoints (Update Based on Actual Documentation)

The script uses placeholder endpoints. Update based on actual HeyGen API:

- **Generate Video**: `POST /v1/video/generate`
- **Check Status**: `GET /v1/video/{video_id}`
- **Download**: `GET /v1/video/{video_id}/download`

## Notes

- Script includes quota tracking to prevent exceeding limits
- Videos are queued for generation (async process)
- Need to poll status and download when ready
- Update script with actual API endpoints and request format




