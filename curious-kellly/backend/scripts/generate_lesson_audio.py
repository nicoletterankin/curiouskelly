#!/usr/bin/env python3
"""
Curious Kellly - Audio Generation Script
Generates multilingual audio files for all lesson variants using ElevenLabs API

Usage:
    python scripts/generate_lesson_audio.py [lesson-id]

Example:
    python scripts/generate_lesson_audio.py the-sun
"""

import os
import sys
import json
import requests
import time
from pathlib import Path

# Get API key from environment
API_KEY = os.getenv('ELEVENLABS_API_KEY')
VOICE_ID = "wAdymQH5YucAkXwmrdL0"  # Kelly voice

# Check API key
if not API_KEY:
    print("❌ ERROR: ELEVENLABS_API_KEY not found")
    print("Add to .env: ELEVENLABS_API_KEY=your_key_here")
    sys.exit(1)

print("✅ ElevenLabs API key loaded")
print(f"🎤 Using voice ID: {VOICE_ID}")
print("\n📝 NOTE: This is a template script.")
print("For full implementation, see AUDIO_GENERATION_PLAN.md")
