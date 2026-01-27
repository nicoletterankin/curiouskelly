#!/usr/bin/env python3
"""
🎬 LOCAL VIDEO GENERATION PIPELINE
=====================================
Generates Kelly lip-sync videos using 100% local processing.

Pipeline:
1. Fetch script from Supabase (lesson_atoms)
2. Generate audio with Tortoise TTS (local)
3. Generate lip-sync video with SadTalker (local)
4. Upload to Supabase kelly-videos bucket
5. Register in lesson_video_generation_status

Hardware Requirements:
- NVIDIA RTX 5090 (32GB VRAM) or similar
- 32GB+ RAM
- CUDA 12.x

Usage:
    python local_video_pipeline.py --day 51 --phase hook --archetype scientist
    python local_video_pipeline.py --day 1 --phase hook --test  # Test with Day 1

Author: Kelly-Sync Pipeline
"""

import os
import sys
import time
import json
import argparse
import logging
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import torch
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('kelly-local-pipeline')

# ============================================================================
# CONFIGURATION
# ============================================================================

SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://tvjalxxsyryjphkforjv.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY', os.environ.get('SUPABASE_SERVICE_KEY', ''))

# Kelly reference image (4K quality)
KELLY_REFERENCE_IMAGE = Path("C:/iLearnStudio/projects/Kelly/Ref/Best Character Reference/head and shoulders without chair.png")
KELLY_FALLBACK_IMAGE = "https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-templates/photorealistic-test/kelly_1765361262640.png"

# Model paths
MODELS_DIR = Path(__file__).parent.parent / "models"
SADTALKER_DIR = MODELS_DIR / "SadTalker"
TORTOISE_DIR = MODELS_DIR / "tortoise-tts"

# Output directories
OUTPUT_DIR = Path(__file__).parent.parent / "output"
TEMP_DIR = Path(tempfile.gettempdir()) / "kelly-sync"


# ============================================================================
# SUPABASE CLIENT
# ============================================================================

class SupabaseClient:
    """Minimal Supabase client for video pipeline."""
    
    def __init__(self, url: str, key: str):
        self.url = url.rstrip('/')
        self.key = key
        self.headers = {
            'apikey': key,
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
        }
    
    def get_lesson_script(self, day_number: int, phase: str, archetype: str) -> Optional[str]:
        """Fetch lesson script from lesson_atoms."""
        import requests
        
        # First get core_lesson_id
        response = requests.get(
            f"{self.url}/rest/v1/core_lessons",
            headers=self.headers,
            params={
                'day_number': f'eq.{day_number}',
                'select': 'id,topic',
                'limit': 1,
            }
        )
        
        if not response.ok or not response.json():
            logger.warning(f"No core_lesson found for day {day_number}")
            return None
        
        lesson = response.json()[0]
        lesson_id = lesson['id']
        topic = lesson['topic']
        logger.info(f"Found lesson: Day {day_number} - {topic}")
        
        # Get lesson atom
        response = requests.get(
            f"{self.url}/rest/v1/lesson_atoms",
            headers=self.headers,
            params={
                'core_lesson_id': f'eq.{lesson_id}',
                'archetype': f'eq.{archetype}',
                'phase': f'eq.{phase}',
                'select': 'content',
                'limit': 1,
            }
        )
        
        if not response.ok or not response.json():
            logger.warning(f"No lesson_atom found for {archetype}/{phase}")
            return None
        
        atom = response.json()[0]
        content = atom.get('content', {})
        script = content.get('script', '')
        
        if script:
            logger.info(f"Script length: {len(script)} chars")
            return script
        
        return None
    
    def upload_video(self, local_path: Path, storage_path: str) -> Optional[str]:
        """Upload video to Supabase storage."""
        import requests
        
        logger.info(f"Uploading to: {storage_path}")
        
        with open(local_path, 'rb') as f:
            response = requests.post(
                f"{self.url}/storage/v1/object/kelly-videos/{storage_path}",
                headers={
                    'apikey': self.key,
                    'Authorization': f'Bearer {self.key}',
                    'Content-Type': 'video/mp4',
                    'x-upsert': 'true',
                },
                data=f,
            )
        
        if response.ok:
            public_url = f"{self.url}/storage/v1/object/public/kelly-videos/{storage_path}"
            logger.info(f"Upload successful: {public_url}")
            return public_url
        else:
            logger.error(f"Upload failed: {response.status_code} - {response.text}")
            return None
    
    def register_video(
        self,
        day_number: int,
        phase: str,
        archetype: str,
        video_url: str,
        status: str = 'completed',
    ):
        """Register video in lesson_video_generation_status."""
        import requests
        
        # Get core_lesson_id
        response = requests.get(
            f"{self.url}/rest/v1/core_lessons",
            headers=self.headers,
            params={'day_number': f'eq.{day_number}', 'select': 'id', 'limit': 1}
        )
        
        if not response.ok or not response.json():
            logger.error(f"Cannot find lesson for day {day_number}")
            return
        
        lesson_id = response.json()[0]['id']
        
        # Map phase names
        phase_map = {
            'hook': 'Hook',
            'fact1': 'Fact1', 'q1': 'Fact1',
            'fact2': 'Fact2', 'q2': 'Fact2',
            'fact3': 'Fact3', 'q3': 'Fact3',
            'wisdom': 'Wisdom',
        }
        db_phase = phase_map.get(phase.lower(), phase.title())
        
        # Upsert record
        response = requests.post(
            f"{self.url}/rest/v1/lesson_video_generation_status",
            headers={**self.headers, 'Prefer': 'resolution=merge-duplicates'},
            json={
                'core_lesson_id': lesson_id,
                'archetype': archetype,
                'phase': db_phase,
                'video_type': 'main',
                'status': status,
                'video_url': video_url,
                'completed_at': datetime.utcnow().isoformat(),
            }
        )
        
        if response.ok:
            logger.info(f"Registered video in database: {status}")
        else:
            logger.warning(f"Registration warning: {response.status_code}")


# ============================================================================
# TORTOISE TTS (LOCAL VOICE SYNTHESIS)
# ============================================================================

class TortoiseTTS:
    """Local voice synthesis using Tortoise TTS."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self._model = None
        self._vocoder = None
        
    def _load_model(self):
        """Load Tortoise TTS model."""
        if self._model is not None:
            return
            
        logger.info("Loading Tortoise TTS...")
        
        try:
            from tortoise.api import TextToSpeech
            from tortoise.utils.audio import load_audio, load_voice
            
            self._model = TextToSpeech()
            logger.info("Tortoise TTS loaded successfully")
            
        except ImportError:
            logger.error("Tortoise TTS not installed!")
            logger.error("Install with: pip install tortoise-tts")
            raise
    
    def generate(
        self,
        text: str,
        output_path: Path,
        voice: str = 'emma',  # Kelly-like voice
        preset: str = 'fast',  # Options: ultra_fast, fast, standard, high_quality
    ) -> Path:
        """Generate audio from text."""
        self._load_model()
        
        logger.info(f"Generating audio ({len(text)} chars)...")
        logger.info(f"  Voice: {voice}, Preset: {preset}")
        
        start = time.time()
        
        from tortoise.utils.audio import load_voice
        
        # Load voice samples if custom
        voice_samples, conditioning_latents = load_voice(voice)
        
        # Generate
        gen = self._model.tts_with_preset(
            text,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            preset=preset,
        )
        
        # Save audio
        import torchaudio
        torchaudio.save(str(output_path), gen.squeeze(0).cpu(), 24000)
        
        elapsed = time.time() - start
        logger.info(f"  Audio generated in {elapsed:.1f}s: {output_path}")
        
        return output_path


# ============================================================================
# PIPER TTS (LIGHTWEIGHT FALLBACK)
# ============================================================================

class PiperTTS:
    """Lightweight local TTS using Piper (fallback)."""
    
    def __init__(self):
        self.piper_path = None
        self._find_piper()
    
    def _find_piper(self):
        """Find Piper executable."""
        candidates = [
            Path("piper/piper.exe"),
            Path("C:/piper/piper.exe"),
            Path(__file__).parent.parent / "piper" / "piper.exe",
        ]
        
        for path in candidates:
            if path.exists():
                self.piper_path = path
                logger.info(f"Found Piper at: {path}")
                return
        
        # Try PATH
        try:
            result = subprocess.run(['piper', '--version'], capture_output=True)
            if result.returncode == 0:
                self.piper_path = 'piper'
                return
        except:
            pass
        
        logger.warning("Piper TTS not found - will use alternative")
    
    def generate(
        self,
        text: str,
        output_path: Path,
        model: str = 'en_US-amy-medium',
    ) -> Optional[Path]:
        """Generate audio using Piper."""
        if self.piper_path is None:
            logger.error("Piper not available")
            return None
        
        logger.info(f"Generating audio with Piper ({len(text)} chars)...")
        
        # Write text to temp file
        text_file = TEMP_DIR / "input.txt"
        text_file.parent.mkdir(parents=True, exist_ok=True)
        text_file.write_text(text, encoding='utf-8')
        
        # Run Piper
        cmd = [
            str(self.piper_path),
            '--model', model,
            '--output_file', str(output_path),
        ]
        
        with open(text_file, 'r') as stdin:
            result = subprocess.run(cmd, stdin=stdin, capture_output=True)
        
        if result.returncode == 0 and output_path.exists():
            logger.info(f"Audio generated: {output_path}")
            return output_path
        else:
            logger.error(f"Piper failed: {result.stderr.decode()}")
            return None


# ============================================================================
# SADTALKER (LOCAL LIP SYNC)
# ============================================================================

class SadTalkerLocal:
    """Local lip-sync video generation using SadTalker."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.sadtalker_path = SADTALKER_DIR
        self._check_installation()
    
    def _check_installation(self):
        """Check if SadTalker is installed."""
        inference_script = self.sadtalker_path / "inference.py"
        
        if not inference_script.exists():
            logger.warning(f"SadTalker not found at {self.sadtalker_path}")
            logger.info("Clone from: https://github.com/OpenTalker/SadTalker")
            self.available = False
        else:
            logger.info(f"SadTalker found at: {self.sadtalker_path}")
            self.available = True
    
    def generate(
        self,
        image_path: Path,
        audio_path: Path,
        output_path: Path,
        enhancer: str = 'gfpgan',  # Options: gfpgan, RestoreFormer
        preprocess: str = 'crop',  # Options: crop, resize, full
        still: bool = False,  # Reduce motion
    ) -> Optional[Path]:
        """Generate lip-sync video."""
        if not self.available:
            logger.error("SadTalker not available")
            return None
        
        logger.info("Generating lip-sync video with SadTalker...")
        logger.info(f"  Image: {image_path}")
        logger.info(f"  Audio: {audio_path}")
        
        start = time.time()
        
        # Prepare output directory
        result_dir = output_path.parent / "sadtalker_results"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Run SadTalker
        cmd = [
            sys.executable,
            str(self.sadtalker_path / "inference.py"),
            '--driven_audio', str(audio_path),
            '--source_image', str(image_path),
            '--result_dir', str(result_dir),
            '--enhancer', enhancer,
            '--preprocess', preprocess,
        ]
        
        if still:
            cmd.append('--still')
        
        # Add CUDA device
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = self.device.split(':')[-1] if ':' in self.device else '0'
        
        logger.info(f"  Running: {' '.join(cmd[:5])}...")
        
        result = subprocess.run(
            cmd,
            cwd=str(self.sadtalker_path),
            env=env,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            logger.error(f"SadTalker failed: {result.stderr}")
            return None
        
        # Find output video
        output_videos = list(result_dir.glob("*.mp4"))
        if not output_videos:
            logger.error("No output video found")
            return None
        
        # Move to final location
        latest_video = max(output_videos, key=lambda p: p.stat().st_mtime)
        import shutil
        shutil.move(str(latest_video), str(output_path))
        
        elapsed = time.time() - start
        logger.info(f"  Video generated in {elapsed:.1f}s: {output_path}")
        
        return output_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class LocalVideoPipeline:
    """Complete local video generation pipeline."""
    
    def __init__(
        self,
        device: str = 'cuda:0',
        tts_engine: str = 'tortoise',  # Options: tortoise, piper
    ):
        self.device = device
        self.tts_engine = tts_engine
        
        # Initialize components
        self.supabase = SupabaseClient(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_KEY else None
        self.sadtalker = SadTalkerLocal(device)
        
        # Initialize TTS
        if tts_engine == 'tortoise':
            self.tts = TortoiseTTS(device)
        else:
            self.tts = PiperTTS()
        
        # Create directories
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  TTS Engine: {tts_engine}")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    def get_kelly_image(self) -> Path:
        """Get Kelly reference image."""
        if KELLY_REFERENCE_IMAGE.exists():
            logger.info(f"Using local Kelly image: {KELLY_REFERENCE_IMAGE}")
            return KELLY_REFERENCE_IMAGE
        
        # Download fallback
        import requests
        fallback_path = TEMP_DIR / "kelly_reference.png"
        
        if not fallback_path.exists():
            logger.info(f"Downloading Kelly image from Supabase...")
            response = requests.get(KELLY_FALLBACK_IMAGE)
            if response.ok:
                fallback_path.write_bytes(response.content)
                logger.info(f"Downloaded to: {fallback_path}")
            else:
                raise RuntimeError("Could not get Kelly reference image")
        
        return fallback_path
    
    def generate(
        self,
        day_number: int,
        phase: str,
        archetype: str,
        script: str = None,
        upload: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a single video.
        
        Args:
            day_number: Lesson day (1-365)
            phase: Phase name (hook, q1, q2, q3, wisdom)
            archetype: Kelly archetype (The Scientist, etc.)
            script: Optional script text (fetches from DB if not provided)
            upload: Whether to upload to Supabase
        
        Returns:
            Dict with paths and status
        """
        start_time = time.time()
        result = {
            'day': day_number,
            'phase': phase,
            'archetype': archetype,
            'status': 'pending',
            'audio_path': None,
            'video_path': None,
            'public_url': None,
        }
        
        logger.info("=" * 60)
        logger.info(f"🎬 GENERATING VIDEO: Day {day_number} / {phase} / {archetype}")
        logger.info("=" * 60)
        
        try:
            # Step 1: Get script
            if script is None:
                if self.supabase:
                    script = self.supabase.get_lesson_script(day_number, phase, archetype)
                
                if not script:
                    # Use placeholder for testing
                    script = f"Welcome to day {day_number}! Today we explore {phase}. This is a test of the local video generation pipeline running on your RTX 5090."
                    logger.warning("Using placeholder script (no DB content)")
            
            logger.info(f"Script: {script[:100]}...")
            
            # Step 2: Generate audio
            audio_path = OUTPUT_DIR / f"day_{day_number}_{phase}_{archetype}_audio.wav"
            
            logger.info("\n📢 Step 1/3: Generating audio...")
            self.tts.generate(script, audio_path)
            result['audio_path'] = str(audio_path)
            
            # Step 3: Get Kelly image
            kelly_image = self.get_kelly_image()
            
            # Step 4: Generate lip-sync video
            video_path = OUTPUT_DIR / f"day_{day_number}_{phase}_{archetype}.mp4"
            
            logger.info("\n🎬 Step 2/3: Generating lip-sync video...")
            self.sadtalker.generate(
                image_path=kelly_image,
                audio_path=audio_path,
                output_path=video_path,
            )
            result['video_path'] = str(video_path)
            
            # Step 5: Upload to Supabase
            if upload and self.supabase and video_path.exists():
                logger.info("\n☁️ Step 3/3: Uploading to Supabase...")
                
                storage_path = f"local-pipeline/day_{day_number:03d}/{phase}_{archetype.lower().replace(' ', '_')}.mp4"
                public_url = self.supabase.upload_video(video_path, storage_path)
                
                if public_url:
                    result['public_url'] = public_url
                    
                    # Register in database
                    self.supabase.register_video(
                        day_number=day_number,
                        phase=phase,
                        archetype=archetype,
                        video_url=public_url,
                    )
            
            result['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)
            raise
        
        # Summary
        elapsed = time.time() - start_time
        result['elapsed_seconds'] = elapsed
        
        logger.info("\n" + "=" * 60)
        logger.info(f"✅ COMPLETE in {elapsed:.1f}s")
        logger.info(f"   Audio: {result['audio_path']}")
        logger.info(f"   Video: {result['video_path']}")
        if result['public_url']:
            logger.info(f"   URL: {result['public_url']}")
        logger.info("=" * 60)
        
        return result


# ============================================================================
# CLI
# ============================================================================

ALL_PHASES = ['hook', 'q1', 'q2', 'q3', 'wisdom']

def main():
    parser = argparse.ArgumentParser(
        description='🎬 Local Video Generation Pipeline for Kelly',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video
  python local_video_pipeline.py --day 51 --phase hook
  
  # All phases for one day
  python local_video_pipeline.py --day 1 --all-phases
  
  # Batch: Days 2-50, all phases
  python local_video_pipeline.py --day 2 --end 50 --all-phases
  
  # Test mode
  python local_video_pipeline.py --test
        """
    )
    
    parser.add_argument('--day', '-d', type=int, required=True,
                        help='Lesson day number (1-365)')
    parser.add_argument('--end', '-e', type=int,
                        help='End day for batch processing (inclusive)')
    parser.add_argument('--phase', '-p', default='hook',
                        choices=['hook', 'q1', 'q2', 'q3', 'wisdom', 'cliff', 'outro'],
                        help='Lesson phase (default: hook)')
    parser.add_argument('--all-phases', action='store_true',
                        help='Generate all 5 phases (hook, q1, q2, q3, wisdom)')
    parser.add_argument('--archetype', '-a', default='The Scientist',
                        help='Kelly archetype (default: The Scientist)')
    
    parser.add_argument('--script', '-s', help='Custom script text')
    parser.add_argument('--no-upload', action='store_true',
                        help='Skip Supabase upload')
    
    parser.add_argument('--tts', default='tortoise',
                        choices=['tortoise', 'piper'],
                        help='TTS engine (default: tortoise)')
    parser.add_argument('--device', default='cuda:0',
                        help='CUDA device (default: cuda:0)')
    
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with Day 1')
    
    args = parser.parse_args()
    
    # Test mode overrides
    if args.test:
        args.day = 1
        args.all_phases = True
        logger.info("🧪 TEST MODE - Using Day 1 with all phases")
    
    # Initialize pipeline
    pipeline = LocalVideoPipeline(
        device=args.device,
        tts_engine=args.tts,
    )
    
    # Determine day range
    start_day = args.day
    end_day = args.end or args.day
    
    # Determine phases
    phases = ALL_PHASES if args.all_phases else [args.phase]
    
    # Calculate total jobs
    total_days = end_day - start_day + 1
    total_jobs = total_days * len(phases)
    
    logger.info("=" * 60)
    logger.info(f"🎬 BATCH VIDEO GENERATION")
    logger.info(f"   Days: {start_day} to {end_day} ({total_days} days)")
    logger.info(f"   Phases: {', '.join(phases)} ({len(phases)} phases)")
    logger.info(f"   Total jobs: {total_jobs}")
    logger.info("=" * 60)
    
    # Track results
    results = {
        'total': total_jobs,
        'completed': 0,
        'failed': 0,
        'videos': [],
    }
    
    job_num = 0
    batch_start = time.time()
    
    for day in range(start_day, end_day + 1):
        for phase in phases:
            job_num += 1
            logger.info(f"\n[{job_num}/{total_jobs}] Day {day} / {phase}")
            
            try:
                result = pipeline.generate(
                    day_number=day,
                    phase=phase,
                    archetype=args.archetype,
                    script=args.script if day == start_day else None,  # Only use custom script for first day
                    upload=not args.no_upload,
                )
                
                if result['status'] == 'completed':
                    results['completed'] += 1
                    results['videos'].append({
                        'day': day,
                        'phase': phase,
                        'url': result.get('public_url'),
                    })
                else:
                    results['failed'] += 1
                    
            except Exception as e:
                logger.error(f"Failed: {e}")
                results['failed'] += 1
    
    # Final summary
    batch_elapsed = time.time() - batch_start
    
    logger.info("\n" + "=" * 60)
    logger.info(f"🎬 BATCH COMPLETE")
    logger.info(f"   Total time: {batch_elapsed / 60:.1f} minutes")
    logger.info(f"   Completed: {results['completed']}/{total_jobs}")
    logger.info(f"   Failed: {results['failed']}/{total_jobs}")
    if results['completed'] > 0:
        avg_time = batch_elapsed / results['completed']
        logger.info(f"   Avg time per video: {avg_time:.1f}s")
    logger.info("=" * 60)
    
    # Output results
    print(json.dumps(results, indent=2))
    
    return 0 if results['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
