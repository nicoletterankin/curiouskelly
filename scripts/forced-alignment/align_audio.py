#!/usr/bin/env python3
"""
Forced Alignment Service for Kelly Lip-Sync

Extracts word-level and phoneme-level timing from ElevenLabs audio.
Uses Montreal Forced Aligner (MFA) for precise alignment.

Requirements:
  pip install montreal-forced-aligner pydub textgrid

Usage:
  python align_audio.py --audio kelly_audio.wav --text "Hello everyone!"
  python align_audio.py --audio kelly_audio.wav --transcript transcript.txt
  python align_audio.py --batch-dir ./audio_files --output ./alignments

Output:
  JSON file with word and phoneme timings suitable for lip-sync
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# ARPAbet to simplified phoneme mapping (for consistency with JS mapping)
ARPABET_NORMALIZE = {
    'AA0': 'AA', 'AA1': 'AA', 'AA2': 'AA',
    'AE0': 'AE', 'AE1': 'AE', 'AE2': 'AE',
    'AH0': 'AH', 'AH1': 'AH', 'AH2': 'AH',
    'AO0': 'AO', 'AO1': 'AO', 'AO2': 'AO',
    'AW0': 'AW', 'AW1': 'AW', 'AW2': 'AW',
    'AY0': 'AY', 'AY1': 'AY', 'AY2': 'AY',
    'EH0': 'EH', 'EH1': 'EH', 'EH2': 'EH',
    'ER0': 'ER', 'ER1': 'ER', 'ER2': 'ER',
    'EY0': 'EY', 'EY1': 'EY', 'EY2': 'EY',
    'IH0': 'IH', 'IH1': 'IH', 'IH2': 'IH',
    'IY0': 'IY', 'IY1': 'IY', 'IY2': 'IY',
    'OW0': 'OW', 'OW1': 'OW', 'OW2': 'OW',
    'OY0': 'OY', 'OY1': 'OY', 'OY2': 'OY',
    'UH0': 'UH', 'UH1': 'UH', 'UH2': 'UH',
    'UW0': 'UW', 'UW1': 'UW', 'UW2': 'UW',
}

# Phoneme to viseme category mapping
PHONEME_TO_VISEME = {
    # Vowels
    'AA': 'A', 'AE': 'A', 'AH': 'A',
    'AO': 'O', 'AW': 'O', 'OW': 'O', 'OY': 'O',
    'EH': 'E', 'EY': 'E',
    'IH': 'I', 'IY': 'I',
    'UH': 'U', 'UW': 'U', 'ER': 'R',
    'AY': 'A',  # Diphthong starting with open
    
    # Consonants
    'P': 'M', 'B': 'M', 'M': 'M',  # Bilabials - lips closed
    'F': 'F', 'V': 'F',  # Labiodentals - teeth on lip
    'TH': 'C', 'DH': 'C',  # Dentals
    'T': 'C', 'D': 'C', 'N': 'C', 'S': 'C', 'Z': 'C',  # Alveolars
    'SH': 'SH', 'ZH': 'SH', 'CH': 'SH', 'JH': 'SH',  # Post-alveolars
    'K': 'C', 'G': 'C', 'NG': 'C', 'HH': 'A',  # Velars/glottal
    'L': 'L', 'R': 'R',  # Liquids
    'W': 'U', 'Y': 'I',  # Glides
    
    # Silence
    'SIL': 'REST', 'SP': 'REST', 'spn': 'REST', '': 'REST',
}


# =============================================================================
# ALIGNMENT METHODS
# =============================================================================

def align_with_mfa(audio_path: str, transcript: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Align audio with transcript using Montreal Forced Aligner.
    
    Args:
        audio_path: Path to WAV audio file
        transcript: Text transcript of the audio
        output_dir: Directory for output files (optional)
        
    Returns:
        Dictionary with word and phoneme alignments
    """
    logger.info(f"Aligning audio with MFA: {audio_path}")
    
    # Create temporary directory for MFA
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare corpus directory structure for MFA
        corpus_dir = Path(temp_dir) / "corpus"
        corpus_dir.mkdir()
        
        # Copy audio file
        audio_name = Path(audio_path).stem
        audio_ext = Path(audio_path).suffix
        target_audio = corpus_dir / f"{audio_name}{audio_ext}"
        
        # Convert to WAV if needed
        if audio_ext.lower() != '.wav':
            target_audio = corpus_dir / f"{audio_name}.wav"
            convert_to_wav(audio_path, str(target_audio))
        else:
            import shutil
            shutil.copy(audio_path, target_audio)
        
        # Create transcript file
        transcript_file = corpus_dir / f"{audio_name}.txt"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        # Output directory
        mfa_output = Path(temp_dir) / "aligned"
        mfa_output.mkdir()
        
        try:
            # Run MFA alignment
            cmd = [
                'mfa', 'align',
                str(corpus_dir),
                'english_us_arpa',  # Dictionary
                'english_us_arpa',  # Acoustic model
                str(mfa_output),
                '--clean',
                '--single_speaker',
            ]
            
            logger.info(f"Running MFA: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"MFA failed: {result.stderr}")
                # Fall back to simple alignment
                return align_simple(audio_path, transcript)
            
            # Parse TextGrid output
            textgrid_path = mfa_output / f"{audio_name}.TextGrid"
            if textgrid_path.exists():
                return parse_textgrid(str(textgrid_path))
            else:
                logger.warning("TextGrid not found, using simple alignment")
                return align_simple(audio_path, transcript)
                
        except subprocess.TimeoutExpired:
            logger.error("MFA timed out")
            return align_simple(audio_path, transcript)
        except FileNotFoundError:
            logger.warning("MFA not installed, using simple alignment")
            return align_simple(audio_path, transcript)


def align_with_gentle(audio_path: str, transcript: str) -> Dict[str, Any]:
    """
    Align audio with transcript using Gentle aligner.
    Gentle must be running as a service on localhost:8765.
    
    Args:
        audio_path: Path to audio file
        transcript: Text transcript
        
    Returns:
        Dictionary with word and phoneme alignments
    """
    import requests
    
    logger.info(f"Aligning audio with Gentle: {audio_path}")
    
    try:
        with open(audio_path, 'rb') as audio_file:
            response = requests.post(
                'http://localhost:8765/transcriptions?async=false',
                files={'audio': audio_file},
                data={'transcript': transcript},
                timeout=120
            )
        
        if response.status_code == 200:
            gentle_result = response.json()
            return convert_gentle_to_standard(gentle_result)
        else:
            logger.error(f"Gentle failed: {response.status_code}")
            return align_simple(audio_path, transcript)
            
    except requests.exceptions.ConnectionError:
        logger.warning("Gentle not available, using simple alignment")
        return align_simple(audio_path, transcript)


def align_simple(audio_path: str, transcript: str) -> Dict[str, Any]:
    """
    Simple word-level alignment based on audio duration and word count.
    Used as fallback when MFA/Gentle are not available.
    
    Args:
        audio_path: Path to audio file
        transcript: Text transcript
        
    Returns:
        Dictionary with estimated word timings
    """
    logger.info("Using simple estimation-based alignment")
    
    # Get audio duration
    duration = get_audio_duration(audio_path)
    
    # Split transcript into words
    words = transcript.split()
    if not words:
        return {'words': [], 'phones': [], 'duration': duration}
    
    # Estimate timing based on word length
    total_chars = sum(len(w) for w in words)
    char_duration = duration / max(total_chars, 1)
    
    word_alignments = []
    phone_alignments = []
    current_time = 0.0
    
    for word in words:
        word_duration = len(word) * char_duration * 1.1  # Slight padding
        word_end = current_time + word_duration
        
        word_alignments.append({
            'word': word,
            'start': round(current_time, 3),
            'end': round(word_end, 3),
            'confidence': 0.5,  # Lower confidence for estimates
        })
        
        # Estimate phonemes from letters (very rough)
        phonemes = estimate_phonemes_from_word(word)
        phone_duration = word_duration / max(len(phonemes), 1)
        
        for i, phoneme in enumerate(phonemes):
            phone_start = current_time + (i * phone_duration)
            phone_end = phone_start + phone_duration
            
            phone_alignments.append({
                'phone': phoneme,
                'start': round(phone_start, 3),
                'end': round(phone_end, 3),
                'word': word,
                'viseme': PHONEME_TO_VISEME.get(phoneme, 'REST'),
            })
        
        current_time = word_end + 0.05  # Small gap between words
    
    return {
        'words': word_alignments,
        'phones': phone_alignments,
        'duration': duration,
        'method': 'simple_estimation',
        'confidence': 0.5,
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_textgrid(textgrid_path: str) -> Dict[str, Any]:
    """Parse TextGrid file from MFA output."""
    try:
        import textgrid
    except ImportError:
        logger.error("textgrid package not installed: pip install textgrid")
        return {'words': [], 'phones': [], 'error': 'textgrid package not installed'}
    
    tg = textgrid.TextGrid.fromFile(textgrid_path)
    
    word_alignments = []
    phone_alignments = []
    
    # Find word and phone tiers
    word_tier = None
    phone_tier = None
    
    for tier in tg.tiers:
        if tier.name.lower() in ['words', 'word']:
            word_tier = tier
        elif tier.name.lower() in ['phones', 'phone']:
            phone_tier = tier
    
    # Parse words
    if word_tier:
        for interval in word_tier:
            if interval.mark and interval.mark.strip():
                word_alignments.append({
                    'word': interval.mark,
                    'start': round(interval.minTime, 3),
                    'end': round(interval.maxTime, 3),
                    'confidence': 1.0,
                })
    
    # Parse phones
    current_word_idx = 0
    if phone_tier:
        for interval in phone_tier:
            phone = interval.mark.strip() if interval.mark else ''
            
            # Normalize phoneme
            normalized = ARPABET_NORMALIZE.get(phone, phone.upper())
            if not normalized:
                normalized = 'SIL'
            
            # Find which word this phone belongs to
            word = ''
            if word_alignments:
                for i, w in enumerate(word_alignments):
                    if w['start'] <= interval.minTime < w['end']:
                        word = w['word']
                        break
            
            phone_alignments.append({
                'phone': normalized,
                'start': round(interval.minTime, 3),
                'end': round(interval.maxTime, 3),
                'word': word,
                'viseme': PHONEME_TO_VISEME.get(normalized, 'REST'),
            })
    
    duration = tg.maxTime if tg.maxTime else 0
    
    return {
        'words': word_alignments,
        'phones': phone_alignments,
        'duration': duration,
        'method': 'mfa',
        'confidence': 0.95,
    }


def convert_gentle_to_standard(gentle_result: Dict) -> Dict[str, Any]:
    """Convert Gentle aligner output to standard format."""
    word_alignments = []
    phone_alignments = []
    
    for word_data in gentle_result.get('words', []):
        if word_data.get('case') == 'success':
            word_alignments.append({
                'word': word_data['word'],
                'start': round(word_data['start'], 3),
                'end': round(word_data['end'], 3),
                'confidence': 0.9,
            })
            
            # Extract phones
            for phone_data in word_data.get('phones', []):
                phone = phone_data['phone'].split('_')[0].upper()
                normalized = ARPABET_NORMALIZE.get(phone, phone)
                
                phone_alignments.append({
                    'phone': normalized,
                    'start': round(phone_data['start'], 3),
                    'end': round(phone_data['start'] + phone_data['duration'], 3),
                    'word': word_data['word'],
                    'viseme': PHONEME_TO_VISEME.get(normalized, 'REST'),
                })
    
    return {
        'words': word_alignments,
        'phones': phone_alignments,
        'duration': gentle_result.get('duration', 0),
        'method': 'gentle',
        'confidence': 0.9,
    }


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0
    except ImportError:
        # Fallback using ffprobe
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())
        except:
            logger.warning("Could not determine audio duration, using estimate")
            return 10.0  # Default estimate


def convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert audio file to WAV format."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format='wav')
    except ImportError:
        # Fallback using ffmpeg
        subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-ar', '16000', '-ac', '1',
            output_path
        ], capture_output=True)


def estimate_phonemes_from_word(word: str) -> List[str]:
    """
    Rough phoneme estimation from spelling.
    This is a very simplified mapping for fallback use.
    """
    # Simple letter-to-phoneme rules (very rough)
    LETTER_TO_PHONEME = {
        'a': 'AE', 'e': 'EH', 'i': 'IH', 'o': 'AA', 'u': 'AH',
        'b': 'B', 'c': 'K', 'd': 'D', 'f': 'F', 'g': 'G',
        'h': 'HH', 'j': 'JH', 'k': 'K', 'l': 'L', 'm': 'M',
        'n': 'N', 'p': 'P', 'q': 'K', 'r': 'R', 's': 'S',
        't': 'T', 'v': 'V', 'w': 'W', 'x': 'K', 'y': 'Y', 'z': 'Z',
    }
    
    phonemes = []
    word = word.lower()
    i = 0
    
    while i < len(word):
        char = word[i]
        
        # Handle digraphs
        if i < len(word) - 1:
            digraph = word[i:i+2]
            if digraph == 'th':
                phonemes.append('TH')
                i += 2
                continue
            elif digraph == 'sh':
                phonemes.append('SH')
                i += 2
                continue
            elif digraph == 'ch':
                phonemes.append('CH')
                i += 2
                continue
            elif digraph == 'ng':
                phonemes.append('NG')
                i += 2
                continue
        
        # Single letter mapping
        if char in LETTER_TO_PHONEME:
            phonemes.append(LETTER_TO_PHONEME[char])
        
        i += 1
    
    return phonemes if phonemes else ['SIL']


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_batch(audio_dir: str, output_dir: str, method: str = 'auto') -> Dict[str, Any]:
    """
    Process multiple audio files in a directory.
    
    Args:
        audio_dir: Directory containing audio files and transcripts
        output_dir: Directory for output JSON files
        method: Alignment method ('mfa', 'gentle', 'simple', 'auto')
        
    Returns:
        Summary of processed files
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'processed': [],
        'failed': [],
        'total': 0,
    }
    
    # Find audio files
    audio_extensions = {'.wav', '.mp3', '.m4a', '.ogg', '.flac'}
    audio_files = [f for f in audio_dir.iterdir() if f.suffix.lower() in audio_extensions]
    
    results['total'] = len(audio_files)
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    for audio_file in audio_files:
        # Look for transcript
        transcript_file = audio_file.with_suffix('.txt')
        if not transcript_file.exists():
            # Try .json with transcript field
            json_file = audio_file.with_suffix('.json')
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    transcript = data.get('transcript', data.get('text', ''))
            else:
                logger.warning(f"No transcript found for {audio_file.name}, skipping")
                results['failed'].append({
                    'file': str(audio_file),
                    'error': 'No transcript found',
                })
                continue
        else:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
        
        try:
            # Perform alignment
            if method == 'mfa':
                alignment = align_with_mfa(str(audio_file), transcript)
            elif method == 'gentle':
                alignment = align_with_gentle(str(audio_file), transcript)
            elif method == 'simple':
                alignment = align_simple(str(audio_file), transcript)
            else:  # auto
                alignment = align_with_mfa(str(audio_file), transcript)
            
            # Add metadata
            alignment['source_audio'] = str(audio_file.name)
            alignment['transcript'] = transcript
            
            # Save output
            output_file = output_dir / f"{audio_file.stem}_alignment.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(alignment, f, indent=2)
            
            results['processed'].append({
                'file': str(audio_file),
                'output': str(output_file),
                'words': len(alignment.get('words', [])),
                'phones': len(alignment.get('phones', [])),
            })
            
            logger.info(f"✓ Processed: {audio_file.name}")
            
        except Exception as e:
            logger.error(f"✗ Failed: {audio_file.name} - {e}")
            results['failed'].append({
                'file': str(audio_file),
                'error': str(e),
            })
    
    return results


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Forced alignment for Kelly lip-sync',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file with inline text
  python align_audio.py --audio kelly.wav --text "Hello everyone!"
  
  # Single file with transcript file
  python align_audio.py --audio kelly.wav --transcript transcript.txt
  
  # Batch processing
  python align_audio.py --batch-dir ./audio_files --output ./alignments
  
  # Use specific aligner
  python align_audio.py --audio kelly.wav --text "Hello" --method gentle
        """
    )
    
    parser.add_argument('--audio', help='Path to audio file')
    parser.add_argument('--text', help='Transcript text (inline)')
    parser.add_argument('--transcript', help='Path to transcript file')
    parser.add_argument('--batch-dir', help='Directory for batch processing')
    parser.add_argument('--output', '-o', help='Output file or directory')
    parser.add_argument('--method', choices=['mfa', 'gentle', 'simple', 'auto'],
                        default='auto', help='Alignment method (default: auto)')
    parser.add_argument('--format', choices=['json', 'textgrid'],
                        default='json', help='Output format (default: json)')
    
    args = parser.parse_args()
    
    # Batch processing
    if args.batch_dir:
        output_dir = args.output or './alignments'
        results = process_batch(args.batch_dir, output_dir, args.method)
        
        print("\n" + "=" * 60)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total files: {results['total']}")
        print(f"Processed: {len(results['processed'])}")
        print(f"Failed: {len(results['failed'])}")
        
        if results['failed']:
            print("\nFailed files:")
            for f in results['failed']:
                print(f"  - {f['file']}: {f['error']}")
        
        return
    
    # Single file processing
    if not args.audio:
        parser.error("--audio is required for single file processing")
    
    # Get transcript
    if args.text:
        transcript = args.text
    elif args.transcript:
        with open(args.transcript, 'r', encoding='utf-8') as f:
            transcript = f.read().strip()
    else:
        parser.error("Either --text or --transcript is required")
    
    # Perform alignment
    if args.method == 'mfa':
        result = align_with_mfa(args.audio, transcript)
    elif args.method == 'gentle':
        result = align_with_gentle(args.audio, transcript)
    elif args.method == 'simple':
        result = align_simple(args.audio, transcript)
    else:  # auto
        result = align_with_mfa(args.audio, transcript)
    
    # Add metadata
    result['source_audio'] = os.path.basename(args.audio)
    result['transcript'] = transcript
    
    # Output
    if args.output:
        output_path = args.output
    else:
        output_path = Path(args.audio).stem + '_alignment.json'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Alignment saved to: {output_path}")
    print(f"  Words: {len(result.get('words', []))}")
    print(f"  Phones: {len(result.get('phones', []))}")
    print(f"  Duration: {result.get('duration', 0):.2f}s")
    print(f"  Method: {result.get('method', 'unknown')}")


if __name__ == '__main__':
    main()

