#!/usr/bin/env python3
"""
HeyGen Video Generation Script
Generates videos from audio files using HeyGen API (Avatar IV)
Mindful of 5 minutes/month quota limit
"""

import json
import os
import sys
import requests
from pathlib import Path
import time
import argparse
from typing import Optional, Dict, List

# HeyGen API Configuration (update with actual credentials)
HEYGEN_API_KEY = os.environ.get("HEYGEN_API_KEY", "")
HEYGEN_BASE_URL = "https://api.heygen.com/v1"  # Update with actual endpoint
HEYGEN_AVATAR_ID = os.environ.get("HEYGEN_AVATAR_ID", "")  # Kelly Avatar IV ID

# Quota tracking
QUOTA_LIMIT_MINUTES = 5.0  # 5 minutes per month
QUOTA_USED_FILE = Path(__file__).parent.parent / "lessons" / ".heygen_quota.json"

# Base directories
LESSONS_DIR = Path(__file__).parent.parent / "lessons"
AUDIO_DIR = LESSONS_DIR / "audio"
VIDEO_OUTPUT_DIR = LESSONS_DIR / "videos"


def load_quota_tracking():
    """Load quota usage tracking"""
    if QUOTA_USED_FILE.exists():
        try:
            with open(QUOTA_USED_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "minutes_used": 0.0,
        "videos_generated": 0,
        "last_reset": None
    }


def save_quota_tracking(data):
    """Save quota usage tracking"""
    QUOTA_USED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(QUOTA_USED_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def check_quota(audio_duration_seconds: float) -> bool:
    """Check if we have enough quota for this video"""
    quota = load_quota_tracking()
    minutes_needed = audio_duration_seconds / 60.0
    total_used = quota["minutes_used"] + minutes_needed
    
    if total_used > QUOTA_LIMIT_MINUTES:
        print(f"  ⚠ Quota exceeded: {total_used:.2f}/{QUOTA_LIMIT_MINUTES} minutes")
        return False
    
    return True


def get_audio_duration(audio_file: Path) -> float:
    """Get duration of audio file in seconds"""
    try:
        # Try using mutagen library if available
        from mutagen.mp3 import MP3
        audio = MP3(str(audio_file))
        return audio.info.length
    except ImportError:
        # Fallback: estimate based on file size (rough approximation)
        # MP3 at 128kbps: ~1MB per minute
        size_mb = audio_file.stat().st_size / (1024 * 1024)
        estimated_minutes = size_mb  # Rough estimate
        return estimated_minutes * 60
    except Exception as e:
        print(f"  ⚠ Could not determine audio duration: {e}")
        return 60.0  # Default to 1 minute


def generate_heygen_video(audio_file: Path, output_file: Path, avatar_id: str = None) -> Optional[str]:
    """
    Generate video using HeyGen API
    
    Returns:
        video_id: HeyGen video ID if successful, None otherwise
    """
    if not HEYGEN_API_KEY:
        print("  ⚠ HeyGen API key not set. Set HEYGEN_API_KEY environment variable.")
        return None
    
    avatar_id = avatar_id or HEYGEN_AVATAR_ID
    if not avatar_id:
        print("  ⚠ HeyGen Avatar ID not set. Set HEYGEN_AVATAR_ID environment variable.")
        return None
    
    # Check quota
    audio_duration = get_audio_duration(audio_file)
    if not check_quota(audio_duration):
        return None
    
    print(f"  🎬 Generating video from: {audio_file.name}")
    print(f"     Audio duration: {audio_duration:.1f}s ({audio_duration/60:.2f} minutes)")
    
    try:
        # Upload audio file
        with open(audio_file, 'rb') as f:
            files = {'audio': (audio_file.name, f, 'audio/mpeg')}
            data = {
                'avatar_id': avatar_id,
                'background': 'white',  # White background for director's chair
                'resolution': '1920x1080',  # 16:9 aspect ratio
            }
            
            headers = {
                'X-API-KEY': HEYGEN_API_KEY
            }
            
            # NOTE: Update this endpoint based on actual HeyGen API documentation
            response = requests.post(
                f"{HEYGEN_BASE_URL}/video/generate",
                headers=headers,
                files=files,
                data=data,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                video_id = result.get('video_id')
                
                # Update quota tracking
                quota = load_quota_tracking()
                quota["minutes_used"] += audio_duration / 60.0
                quota["videos_generated"] += 1
                save_quota_tracking(quota)
                
                print(f"  ✅ Video generation started. Video ID: {video_id}")
                print(f"     Quota used: {quota['minutes_used']:.2f}/{QUOTA_LIMIT_MINUTES} minutes")
                
                return video_id
            else:
                print(f"  ❌ Error: {response.status_code} - {response.text[:200]}")
                return None
                
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return None


def check_video_status(video_id: str) -> Dict:
    """Check status of video generation"""
    if not HEYGEN_API_KEY:
        return {"status": "error", "message": "API key not set"}
    
    try:
        headers = {'X-API-KEY': HEYGEN_API_KEY}
        response = requests.get(
            f"{HEYGEN_BASE_URL}/video/{video_id}",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def download_video(video_id: str, output_file: Path) -> bool:
    """Download completed video from HeyGen"""
    if not HEYGEN_API_KEY:
        return False
    
    try:
        headers = {'X-API-KEY': HEYGEN_API_KEY}
        response = requests.get(
            f"{HEYGEN_BASE_URL}/video/{video_id}/download",
            headers=headers,
            stream=True,
            timeout=300
        )
        
        if response.status_code == 200:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  ✅ Video downloaded: {output_file.name}")
            return True
        else:
            print(f"  ❌ Download error: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Download exception: {e}")
        return False


def generate_lesson_videos(lesson_id: str, audio_dir: Path, output_dir: Path, 
                          priority_phases: List[str] = None, max_videos: int = None):
    """Generate videos for a lesson, prioritizing certain phases"""
    
    priority_phases = priority_phases or ["welcome", "mainContent"]
    lesson_audio_dir = audio_dir / lesson_id
    
    if not lesson_audio_dir.exists():
        print(f"  ⚠ Audio directory not found: {lesson_audio_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Generating Videos for: {lesson_id}")
    print(f"{'='*60}")
    
    # Find all audio files
    audio_files = []
    for audio_file in lesson_audio_dir.glob("*.mp3"):
        # Parse filename: age-language-phase.mp3
        parts = audio_file.stem.split('-')
        if len(parts) >= 3:
            age = parts[0]
            language = parts[1]
            phase = '-'.join(parts[2:])  # Handle phases with hyphens
            
            priority = 1 if phase in priority_phases else 2
            audio_files.append({
                "file": audio_file,
                "age": age,
                "language": language,
                "phase": phase,
                "priority": priority
            })
    
    # Sort by priority, then by age/language
    audio_files.sort(key=lambda x: (x["priority"], x["age"], x["language"]))
    
    # Limit number of videos if specified
    if max_videos:
        audio_files = audio_files[:max_videos]
    
    quota = load_quota_tracking()
    remaining_minutes = QUOTA_LIMIT_MINUTES - quota["minutes_used"]
    
    print(f"  Quota remaining: {remaining_minutes:.2f}/{QUOTA_LIMIT_MINUTES} minutes")
    print(f"  Audio files found: {len(audio_files)}")
    print(f"  Will generate: {len(audio_files)} videos")
    print()
    
    generated = 0
    skipped = 0
    
    for audio_info in audio_files:
        audio_file = audio_info["file"]
        age = audio_info["age"]
        language = audio_info["language"]
        phase = audio_info["phase"]
        
        # Check quota before generating
        audio_duration = get_audio_duration(audio_file)
        if not check_quota(audio_duration):
            print(f"  ⏭ Skipping {audio_file.name} (quota exceeded)")
            skipped += 1
            continue
        
        # Generate video
        output_file = output_dir / lesson_id / f"{age}-{language}-{phase}.mp4"
        video_id = generate_heygen_video(audio_file, output_file)
        
        if video_id:
            generated += 1
            print(f"  → Video ID: {video_id} (will need to check status and download)")
        else:
            skipped += 1
        
        # Rate limiting
        time.sleep(2)
    
    print(f"\n  Summary: {generated} videos queued, {skipped} skipped")
    quota = load_quota_tracking()
    print(f"  Quota used: {quota['minutes_used']:.2f}/{QUOTA_LIMIT_MINUTES} minutes")


def main():
    parser = argparse.ArgumentParser(description="Generate videos using HeyGen API")
    parser.add_argument("--lesson", help="Generate videos for specific lesson ID")
    parser.add_argument("--audio-dir", default=AUDIO_DIR, help="Directory containing audio files")
    parser.add_argument("--output-dir", default=VIDEO_OUTPUT_DIR, help="Output directory for videos")
    parser.add_argument("--priority-phases", nargs="+", default=["welcome", "mainContent"],
                       help="Priority phases to generate first")
    parser.add_argument("--max-videos", type=int, help="Maximum number of videos to generate")
    parser.add_argument("--check-status", help="Check status of video ID")
    parser.add_argument("--download", help="Download video by ID")
    parser.add_argument("--download-to", help="Output path for download")
    
    args = parser.parse_args()
    
    print("="*60)
    print("HeyGen Video Generation")
    print("="*60)
    print(f"API Key: {'*' * 20}...{HEYGEN_API_KEY[-10:] if len(HEYGEN_API_KEY) > 10 else 'NOT SET'}")
    print(f"Avatar ID: {HEYGEN_AVATAR_ID or 'NOT SET'}")
    print(f"Quota Limit: {QUOTA_LIMIT_MINUTES} minutes/month")
    
    quota = load_quota_tracking()
    print(f"Quota Used: {quota['minutes_used']:.2f} minutes")
    print(f"Videos Generated: {quota['videos_generated']}")
    print("="*60)
    
    # Check status
    if args.check_status:
        status = check_video_status(args.check_status)
        print(f"Video Status: {json.dumps(status, indent=2)}")
        return 0
    
    # Download video
    if args.download:
        output_path = Path(args.download_to) if args.download_to else VIDEO_OUTPUT_DIR / f"{args.download}.mp4"
        success = download_video(args.download, output_path)
        return 0 if success else 1
    
    # Generate videos
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    
    if args.lesson:
        # Generate for single lesson
        generate_lesson_videos(
            args.lesson,
            audio_dir,
            output_dir,
            args.priority_phases,
            args.max_videos
        )
    else:
        # Generate for all lessons (with quota limits)
        print("\n⚠ Generating videos for all lessons with quota limits")
        print("  Consider using --lesson to generate specific lessons")
        print("  Use --max-videos to limit generation")
        
        lesson_dirs = [d for d in audio_dir.iterdir() if d.is_dir()]
        print(f"\nFound {len(lesson_dirs)} lessons")
        
        for lesson_dir in lesson_dirs[:3]:  # Limit to first 3 lessons by default
            lesson_id = lesson_dir.name
            generate_lesson_videos(
                lesson_id,
                audio_dir,
                output_dir,
                args.priority_phases,
                args.max_videos or 2  # Default to 2 videos per lesson
            )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())




