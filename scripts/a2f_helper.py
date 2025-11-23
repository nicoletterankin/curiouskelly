#!/usr/bin/env python3
"""
Helper script for Audio2Face pipeline:
1. Convert MP3 to WAV (16-bit PCM, mono, 44100 Hz)
2. Convert Audio2Face CSV output to Unity JSON format
"""

import sys
import json
import csv
import os
from pathlib import Path

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

def mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 to WAV format required by Audio2Face (16-bit PCM, mono, 44100 Hz)"""
    if not HAS_PYDUB:
        print("ERROR: pydub not installed. Install with: pip install pydub")
        print("Note: On Windows, also install ffmpeg or use: pip install pydub[mp3]")
        return False
    
    try:
        # Load MP3
        audio = AudioSegment.from_mp3(mp3_path)
        
        # Convert to mono, 44100 Hz, 16-bit PCM
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(44100)
        audio = audio.set_sample_width(2)  # 16-bit = 2 bytes
        
        # Export as WAV
        audio.export(wav_path, format="wav")
        return True
    except FileNotFoundError as e:
        if "ffmpeg" in str(e).lower() or "avconv" in str(e).lower():
            print("ERROR: ffmpeg not found. pydub requires ffmpeg for MP3 conversion.")
            print("")
            print("Install ffmpeg:")
            print("  1. Download from: https://ffmpeg.org/download.html")
            print("  2. Extract and add to Windows PATH")
            print("  3. Or use: winget install ffmpeg")
            print("")
            print("Alternative: Use WAV files directly (skip MP3 conversion)")
        else:
            print(f"ERROR: File not found: {e}")
        return False
    except Exception as e:
        print(f"ERROR converting MP3 to WAV: {e}")
        if "ffmpeg" in str(e).lower():
            print("")
            print("ffmpeg is required for MP3 conversion. Install from: https://ffmpeg.org/download.html")
        return False

def csv_to_unity_json(csv_path, json_path, fps=30):
    """
    Convert Audio2Face CSV output to Unity JSON format.
    
    CSV format: timeCode, blendShapes.EyeBlinkLeft, blendShapes.EyeBlinkRight, ...
    Unity format: { "fps": 30, "frames": [{"EyeBlinkLeft": 0.5, ...}, ...] }
    """
    try:
        frames = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Extract blendshape values (columns starting with "blendShapes.")
                frame_data = {}
                
                for key, value in row.items():
                    if key.startswith("blendShapes."):
                        # Remove "blendShapes." prefix
                        blendshape_name = key.replace("blendShapes.", "")
                        try:
                            # Convert to float, handle empty strings
                            weight = float(value) if value.strip() else 0.0
                            frame_data[blendshape_name] = weight
                        except (ValueError, TypeError):
                            frame_data[blendshape_name] = 0.0
                
                if frame_data:  # Only add non-empty frames
                    frames.append(frame_data)
        
        # Create Unity JSON structure
        unity_data = {
            "fps": fps,
            "frames": frames
        }
        
        # Write JSON file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(unity_data, f, indent=2)
        
        print(f"✓ Converted {len(frames)} frames to Unity JSON format")
        return True
        
    except Exception as e:
        print(f"ERROR converting CSV to JSON: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python a2f_helper.py mp3-to-wav <input.mp3> <output.wav>")
        print("  python a2f_helper.py csv-to-json <input.csv> <output.json> [fps]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "mp3-to-wav":
        if len(sys.argv) != 4:
            print("Usage: python a2f_helper.py mp3-to-wav <input.mp3> <output.wav>")
            sys.exit(1)
        
        mp3_path = sys.argv[2]
        wav_path = sys.argv[3]
        
        if not os.path.exists(mp3_path):
            print(f"ERROR: Input file not found: {mp3_path}")
            sys.exit(1)
        
        success = mp3_to_wav(mp3_path, wav_path)
        sys.exit(0 if success else 1)
    
    elif command == "csv-to-json":
        if len(sys.argv) < 4:
            print("Usage: python a2f_helper.py csv-to-json <input.csv> <output.json> [fps]")
            sys.exit(1)
        
        csv_path = sys.argv[2]
        json_path = sys.argv[3]
        fps = int(sys.argv[4]) if len(sys.argv) > 4 else 30
        
        if not os.path.exists(csv_path):
            print(f"ERROR: Input file not found: {csv_path}")
            sys.exit(1)
        
        success = csv_to_unity_json(csv_path, json_path, fps)
        sys.exit(0 if success else 1)
    
    else:
        print(f"ERROR: Unknown command: {command}")
        sys.exit(1)

