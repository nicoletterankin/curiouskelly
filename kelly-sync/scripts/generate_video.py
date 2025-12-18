#!/usr/bin/env python3
"""
🎬 KELLY-SYNC VIDEO GENERATOR

Main entry point for generating high-quality Kelly videos locally.

This script orchestrates the full pipeline:
1. Generate audio from script (ElevenLabs)
2. Synthesize lip-synced video (VideoReTalking)
3. Restore face quality (CodeFormer)
4. Upscale to 4K/8K (Real-ESRGAN)
5. Apply motion templates (FOMM)
6. Export final video (FFmpeg)

Usage:
    # Single video from audio
    python generate_video.py --audio lesson.mp3 --output kelly_lesson.mp4

    # Generate from script text
    python generate_video.py --script "Hello, welcome to today's lesson!" --output kelly_hello.mp4

    # Full day generation (all archetypes)
    python generate_video.py --day 352 --all-archetypes

    # Specific archetype
    python generate_video.py --day 352 --archetype scientist

Quality presets:
    --quality draft     # Fast, 720p, no super-res (testing)
    --quality standard  # 1080p with restoration (daily use)
    --quality premium   # 4K with full pipeline (production)
    --quality ultra     # 8K, maximum quality (archive)
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('kelly-sync')

# Quality presets
QUALITY_PRESETS = {
    'draft': {
        'resolution': '720p',
        'enable_restoration': False,
        'enable_super_resolution': False,
        'enable_motion_transfer': False,
        'codeformer_fidelity': 0.9,
        'crf': 23,
    },
    'standard': {
        'resolution': '1080p',
        'enable_restoration': True,
        'enable_super_resolution': False,
        'enable_motion_transfer': True,
        'codeformer_fidelity': 0.7,
        'crf': 20,
    },
    'premium': {
        'resolution': '4k',
        'enable_restoration': True,
        'enable_super_resolution': True,
        'enable_motion_transfer': True,
        'codeformer_fidelity': 0.6,
        'crf': 18,
    },
    'ultra': {
        'resolution': '8k',
        'enable_restoration': True,
        'enable_super_resolution': True,
        'enable_motion_transfer': True,
        'codeformer_fidelity': 0.5,
        'crf': 15,
    },
}

# Kelly archetypes
ARCHETYPES = [
    'scientist', 'explorer', 'rebel', 'architect', 'diplomat', 'empath',
    'macgyver', 'mystic', 'provider', 'storyteller', 'strategist', 'survivor'
]


class KellyVideoGenerator:
    """
    High-level interface for Kelly video generation.
    
    Wraps the full pipeline with sensible defaults and
    handles audio generation, motion templates, and batch processing.
    """
    
    def __init__(
        self,
        config_path: str = None,
        quality: str = 'premium',
        device: str = 'cuda:0',
    ):
        self.quality = quality
        self.quality_preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS['premium'])
        self.device = device
        
        # Load configuration
        config_path = config_path or Path(__file__).parent.parent / 'config.yaml'
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Paths
        self.kelly_reference = Path(self.config['kelly']['reference_image'])
        self.motion_library = Path(self.config['kelly']['motion_library'])
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # API keys (from environment)
        self.elevenlabs_key = os.environ.get('ELEVENLABS_API_KEY')
        self.kelly_voice_id = self.config['kelly']['voice_id']
        
        # Pipeline components (lazy loaded)
        self._pipeline = None
        self._audio_generator = None
        
        logger.info(f"KellyVideoGenerator initialized")
        logger.info(f"  Quality: {quality}")
        logger.info(f"  Resolution: {self.quality_preset['resolution']}")
        logger.info(f"  Device: {device}")
    
    def generate_audio(
        self,
        script: str,
        output_path: str = None,
    ) -> str:
        """
        Generate Kelly voice audio from script text.
        
        Args:
            script: Text to speak
            output_path: Where to save audio (optional)
        
        Returns:
            Path to generated audio file
        """
        import requests
        
        if not self.elevenlabs_key:
            raise ValueError("ELEVENLABS_API_KEY not set in environment")
        
        logger.info("Generating audio with ElevenLabs...")
        logger.info(f"  Script length: {len(script)} chars")
        
        # ElevenLabs API
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.kelly_voice_id}"
        
        response = requests.post(
            url,
            headers={
                'xi-api-key': self.elevenlabs_key,
                'Content-Type': 'application/json',
            },
            json={
                'text': script,
                'model_id': 'eleven_multilingual_v2',
                'voice_settings': {
                    'stability': 0.5,
                    'similarity_boost': 0.85,
                    'style': 0.3,
                    'use_speaker_boost': True,
                },
            },
        )
        
        if not response.ok:
            raise RuntimeError(f"ElevenLabs API error: {response.status_code} - {response.text}")
        
        # Save audio
        if output_path is None:
            output_path = self.output_dir / f"audio_{int(time.time())}.mp3"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"  Audio saved: {output_path}")
        
        return str(output_path)
    
    def generate(
        self,
        audio_path: str = None,
        script: str = None,
        archetype: str = 'scientist',
        output_path: str = None,
        motion_template: str = None,
    ) -> str:
        """
        Generate a Kelly video.
        
        Args:
            audio_path: Path to audio file (or use script)
            script: Text script (generates audio if no audio_path)
            archetype: Kelly archetype for motion template
            output_path: Output video path
            motion_template: Explicit motion template video
        
        Returns:
            Path to generated video
        """
        start_time = time.time()
        
        logger.info("="*60)
        logger.info("🎬 KELLY VIDEO GENERATION")
        logger.info("="*60)
        
        # Generate audio if needed
        if audio_path is None:
            if script is None:
                raise ValueError("Must provide either audio_path or script")
            audio_path = self.generate_audio(script)
        
        # Determine output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"kelly_{archetype}_{timestamp}.mp4"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load motion template for archetype
        if motion_template is None:
            motion_template = self._get_motion_template(archetype)
        
        logger.info(f"  Audio: {audio_path}")
        logger.info(f"  Archetype: {archetype}")
        logger.info(f"  Output: {output_path}")
        
        # Import and run pipeline
        from src.pipeline import KellySyncPipeline, PipelineConfig
        
        config = PipelineConfig(
            audio_path=str(audio_path),
            reference_image=str(self.kelly_reference),
            output_path=str(output_path),
            resolution=self.quality_preset['resolution'],
            motion_template=motion_template,
            codeformer_fidelity=self.quality_preset['codeformer_fidelity'],
            enable_super_resolution=self.quality_preset['enable_super_resolution'],
            enable_motion_transfer=self.quality_preset['enable_motion_transfer'],
            device=self.device,
        )
        
        pipeline = KellySyncPipeline(config)
        result = pipeline.run()
        
        # Log completion
        elapsed = time.time() - start_time
        logger.info("="*60)
        logger.info(f"✅ Video generated in {elapsed:.1f}s")
        logger.info(f"   Output: {result}")
        logger.info("="*60)
        
        return result
    
    def _get_motion_template(self, archetype: str) -> Optional[str]:
        """Get motion template path for archetype from HeyGen archive."""
        # Check for archived HeyGen video
        archive_patterns = [
            f"generated-videos/heygen-archive/day-351-{archetype}-*.mp4",
            f"assets/motion_templates/{archetype}/motion_B.mp4",
        ]
        
        for pattern in archive_patterns:
            import glob
            matches = glob.glob(pattern)
            if matches:
                logger.info(f"  Motion template: {matches[0]}")
                return matches[0]
        
        logger.warning(f"  No motion template found for {archetype}")
        return None
    
    def generate_day(
        self,
        day_number: int,
        archetypes: List[str] = None,
        scripts_path: str = None,
    ) -> Dict[str, str]:
        """
        Generate videos for all archetypes for a given day.
        
        Args:
            day_number: Lesson day number (1-365)
            archetypes: List of archetypes (default: all 12)
            scripts_path: Path to lesson scripts JSON
        
        Returns:
            Dict mapping archetype to output path
        """
        archetypes = archetypes or ARCHETYPES
        
        logger.info("="*60)
        logger.info(f"🎬 GENERATING DAY {day_number} VIDEOS")
        logger.info(f"   Archetypes: {len(archetypes)}")
        logger.info("="*60)
        
        # Load lesson script
        script = self._load_lesson_script(day_number, scripts_path)
        
        results = {}
        
        for i, archetype in enumerate(archetypes):
            logger.info(f"\n[{i+1}/{len(archetypes)}] {archetype.upper()}")
            
            try:
                output_path = self.output_dir / f"day-{day_number}" / f"{archetype}.mp4"
                
                result = self.generate(
                    script=script,
                    archetype=archetype,
                    output_path=str(output_path),
                )
                
                results[archetype] = result
                logger.info(f"  ✅ {archetype}: {result}")
                
            except Exception as e:
                logger.error(f"  ❌ {archetype}: {e}")
                results[archetype] = None
        
        # Summary
        successful = sum(1 for v in results.values() if v)
        logger.info("\n" + "="*60)
        logger.info(f"DAY {day_number} COMPLETE: {successful}/{len(archetypes)} videos")
        logger.info("="*60)
        
        # Save manifest
        manifest_path = self.output_dir / f"day-{day_number}" / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump({
                'day': day_number,
                'generated': datetime.now().isoformat(),
                'quality': self.quality,
                'videos': results,
            }, f, indent=2)
        
        return results
    
    def _load_lesson_script(self, day_number: int, scripts_path: str = None) -> str:
        """Load lesson script for a given day."""
        # Try to load from lessons JSON
        lesson_paths = [
            scripts_path,
            f"public/lessons/day-{day_number}.json",
            f"content/lessons/day-{day_number}.json",
        ]
        
        for path in lesson_paths:
            if path and Path(path).exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    # Combine all phase scripts
                    phases = ['hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom', 'outro']
                    scripts = [data.get(p, {}).get('script', '') for p in phases]
                    return ' '.join(s for s in scripts if s)
        
        # Fallback: use a placeholder
        logger.warning(f"No lesson script found for day {day_number}, using placeholder")
        return f"Welcome to day {day_number}. This is a placeholder script for testing the video generation pipeline."


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='🎬 Kelly Video Generator - Production-quality local video pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from audio file
  python generate_video.py --audio lesson.mp3 -o output.mp4

  # Generate from script text
  python generate_video.py --script "Hello world!" -o hello.mp4

  # Generate full day (all 12 archetypes)
  python generate_video.py --day 352 --all-archetypes

  # Specific archetype with quality preset
  python generate_video.py --day 352 --archetype scientist --quality premium

Quality presets:
  draft    - 720p, fast, for testing
  standard - 1080p, balanced
  premium  - 4K, full pipeline (default)
  ultra    - 8K, maximum quality
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--audio', '-a', help='Path to audio file')
    input_group.add_argument('--script', '-s', help='Text script to speak')
    input_group.add_argument('--day', '-d', type=int, help='Generate videos for lesson day (1-365)')
    
    # Output options
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--output-dir', default='output', help='Output directory for batch generation')
    
    # Generation options
    parser.add_argument('--archetype', choices=ARCHETYPES, default='scientist',
                        help='Kelly archetype (default: scientist)')
    parser.add_argument('--all-archetypes', action='store_true',
                        help='Generate for all 12 archetypes')
    parser.add_argument('--quality', '-q', choices=['draft', 'standard', 'premium', 'ultra'],
                        default='premium', help='Quality preset (default: premium)')
    
    # Technical options
    parser.add_argument('--config', help='Path to config.yaml')
    parser.add_argument('--device', default='cuda:0', help='Device for processing')
    parser.add_argument('--motion-template', help='Path to motion template video')
    
    # Utility options
    parser.add_argument('--list-archetypes', action='store_true', help='List available archetypes')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be generated')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle utility commands
    if args.list_archetypes:
        print("Available Kelly archetypes:")
        for arch in ARCHETYPES:
            print(f"  - {arch}")
        return 0
    
    # Validate input
    if not any([args.audio, args.script, args.day]):
        parser.error("Must provide --audio, --script, or --day")
    
    # Initialize generator
    generator = KellyVideoGenerator(
        config_path=args.config,
        quality=args.quality,
        device=args.device,
    )
    
    # Dry run
    if args.dry_run:
        print("\n🔍 DRY RUN - Would generate:")
        if args.day:
            archetypes = ARCHETYPES if args.all_archetypes else [args.archetype]
            print(f"  Day: {args.day}")
            print(f"  Archetypes: {len(archetypes)}")
            for arch in archetypes:
                print(f"    - {arch}")
            print(f"  Quality: {args.quality}")
            print(f"  Resolution: {QUALITY_PRESETS[args.quality]['resolution']}")
        else:
            print(f"  Archetype: {args.archetype}")
            print(f"  Output: {args.output or 'auto'}")
            print(f"  Quality: {args.quality}")
        return 0
    
    # Generate
    try:
        if args.day:
            # Day generation
            archetypes = ARCHETYPES if args.all_archetypes else [args.archetype]
            results = generator.generate_day(
                day_number=args.day,
                archetypes=archetypes,
            )
            
            # Print summary
            successful = sum(1 for v in results.values() if v)
            print(f"\n✅ Generated {successful}/{len(archetypes)} videos")
            
        else:
            # Single video generation
            result = generator.generate(
                audio_path=args.audio,
                script=args.script,
                archetype=args.archetype,
                output_path=args.output,
                motion_template=args.motion_template,
            )
            
            print(f"\n✅ Video generated: {result}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
