#!/usr/bin/env python3
"""
🎤 KELLY-SYNC Audio Processor

Extracts phonemes and audio features for lip synchronization.
Uses OpenAI Whisper for accurate phoneme-level timing.
"""

import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch
import librosa

logger = logging.getLogger('kelly-sync.audio')

# Phoneme to Viseme mapping (CMU ARPABET to mouth shape)
PHONEME_TO_VISEME = {
    # Silence
    'sil': 0, 'sp': 0, 'spn': 0,
    
    # Bilabials (lips together) - /p/, /b/, /m/
    'P': 1, 'B': 1, 'M': 1,
    
    # Labiodentals (teeth on lip) - /f/, /v/
    'F': 2, 'V': 2,
    
    # Interdentals (tongue between teeth) - /th/
    'TH': 3, 'DH': 3,
    
    # Alveolars (tongue behind teeth) - /t/, /d/, /n/, /l/
    'T': 4, 'D': 4, 'N': 4, 'L': 4,
    
    # Palatals (tongue on palate) - /sh/, /ch/, /j/
    'SH': 5, 'CH': 5, 'JH': 5, 'ZH': 5,
    
    # Velars (back of tongue) - /k/, /g/, /ng/
    'K': 6, 'G': 6, 'NG': 6,
    
    # Sibilants - /s/, /z/
    'S': 7, 'Z': 7,
    
    # Vowels - open mouth shapes
    'AA': 8, 'AE': 8, 'AH': 8, 'AO': 8, 'AW': 8,  # Open A
    'EH': 9, 'ER': 9, 'EY': 9,                     # E sounds
    'IH': 10, 'IY': 10,                            # I sounds
    'OW': 11, 'OY': 11,                            # O sounds
    'UH': 12, 'UW': 12,                            # U sounds
    
    # R sound (rounded)
    'R': 13,
    
    # W/Y (semi-vowels)
    'W': 14, 'Y': 14, 'HH': 14,
}


class AudioProcessor:
    """
    Processes audio for lip synchronization.
    
    Features:
    - Whisper-based phoneme extraction
    - Mel spectrogram for VideoReTalking
    - Precise timing alignment
    """
    
    def __init__(
        self,
        device: torch.device = None,
        whisper_model: str = "large-v3",
        sample_rate: int = 16000,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.whisper_model_name = whisper_model
        self.sample_rate = sample_rate
        self._whisper = None
        
        logger.info(f"AudioProcessor initialized on {self.device}")
    
    @property
    def whisper(self):
        """Lazy load Whisper model."""
        if self._whisper is None:
            import whisper
            logger.info(f"Loading Whisper {self.whisper_model_name}...")
            self._whisper = whisper.load_model(self.whisper_model_name, device=self.device)
        return self._whisper
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sample rate.
        
        Returns:
            Tuple of (waveform, sample_rate)
        """
        logger.info(f"Loading audio: {audio_path}")
        
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        duration = len(waveform) / sr
        logger.info(f"  Duration: {duration:.2f}s, Sample rate: {sr}Hz")
        
        return waveform, sr
    
    def extract_phonemes(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Extract phoneme-level alignments using Whisper.
        
        Returns:
            List of phoneme segments with timing:
            [{"phoneme": "P", "start": 0.0, "end": 0.1, "viseme": 1}, ...]
        """
        logger.info("Extracting phonemes with Whisper...")
        
        # Transcribe with word-level timestamps
        result = self.whisper.transcribe(
            audio_path,
            word_timestamps=True,
            task="transcribe",
        )
        
        # Extract word-level timing (Whisper doesn't give phoneme-level directly)
        # We'll estimate phoneme timing from word timing
        phonemes = []
        
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                word = word_info.get("word", "").strip()
                start = word_info.get("start", 0)
                end = word_info.get("end", 0)
                
                if not word:
                    continue
                
                # Distribute phonemes evenly across word duration
                # In production, use a proper G2P (grapheme-to-phoneme) model
                word_phonemes = self._estimate_phonemes(word)
                word_duration = end - start
                phoneme_duration = word_duration / max(len(word_phonemes), 1)
                
                for i, p in enumerate(word_phonemes):
                    viseme = PHONEME_TO_VISEME.get(p.upper(), 0)
                    phonemes.append({
                        "phoneme": p,
                        "start": start + i * phoneme_duration,
                        "end": start + (i + 1) * phoneme_duration,
                        "viseme": viseme,
                    })
        
        logger.info(f"  Extracted {len(phonemes)} phoneme segments")
        return phonemes
    
    def _estimate_phonemes(self, word: str) -> List[str]:
        """
        Simple phoneme estimation from text.
        In production, replace with CMU Pronouncing Dictionary or G2P model.
        """
        # Very simplified - just return letter-based approximation
        # Real implementation should use cmudict or g2p-en
        phonemes = []
        word_lower = word.lower()
        
        # Basic phoneme rules
        i = 0
        while i < len(word_lower):
            c = word_lower[i]
            
            # Check for digraphs
            if i < len(word_lower) - 1:
                digraph = word_lower[i:i+2]
                if digraph in ['th', 'ch', 'sh', 'ng', 'wh']:
                    phonemes.append(digraph.upper())
                    i += 2
                    continue
            
            # Single characters
            if c in 'aeiou':
                phonemes.append('AH' if c == 'a' else 
                               'EH' if c == 'e' else
                               'IH' if c == 'i' else
                               'OW' if c == 'o' else 'UW')
            elif c in 'bcdfghjklmnpqrstvwxyz':
                phonemes.append(c.upper())
            
            i += 1
        
        return phonemes if phonemes else ['sil']
    
    def extract_mel_spectrogram(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        n_mels: int = 80,
        hop_length: int = 160,
    ) -> np.ndarray:
        """
        Extract mel spectrogram for VideoReTalking input.
        
        Args:
            waveform: Audio waveform
            sample_rate: Sample rate
            n_mels: Number of mel bands
            hop_length: Hop length in samples (160 = 10ms at 16kHz)
        
        Returns:
            Mel spectrogram of shape (n_mels, time_frames)
        """
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            fmin=0,
            fmax=8000,
        )
        
        # Convert to log scale
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize to [-1, 1]
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min()) * 2 - 1
        
        return mel_norm
    
    def process(self, audio_path: str) -> Tuple[List[Dict], np.ndarray]:
        """
        Full audio processing pipeline.
        
        Returns:
            Tuple of (phonemes, audio_features)
        """
        # Load audio
        waveform, sr = self.load_audio(audio_path)
        
        # Extract phonemes
        phonemes = self.extract_phonemes(audio_path)
        
        # Extract mel spectrogram for lip sync
        mel = self.extract_mel_spectrogram(waveform, sr)
        
        logger.info(f"  Audio features shape: {mel.shape}")
        
        return phonemes, mel
    
    def get_viseme_sequence(
        self,
        phonemes: List[Dict],
        fps: int = 30,
        total_frames: int = None,
    ) -> np.ndarray:
        """
        Convert phoneme sequence to per-frame viseme indices.
        
        Args:
            phonemes: List of phoneme dicts with timing
            fps: Target frame rate
            total_frames: Total number of output frames
        
        Returns:
            Array of viseme indices, one per frame
        """
        if not phonemes:
            return np.zeros(total_frames or 30, dtype=np.int32)
        
        # Calculate total duration
        max_time = max(p['end'] for p in phonemes)
        total_frames = total_frames or int(max_time * fps)
        
        visemes = np.zeros(total_frames, dtype=np.int32)
        
        for phoneme in phonemes:
            start_frame = int(phoneme['start'] * fps)
            end_frame = int(phoneme['end'] * fps)
            viseme = phoneme.get('viseme', 0)
            
            visemes[start_frame:end_frame] = viseme
        
        return visemes
