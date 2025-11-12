"""
Voice manipulation utilities for the hybrid TTS system.

This module provides utility functions for voice manipulation, conversion,
and quality assessment.
"""

import torch
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.signal import resample
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path
import json


class VoiceUtils:
    """
    Utility class for voice manipulation and analysis.
    
    This class provides various utility functions for working with
    voice embeddings, audio signals, and voice characteristics.
    """
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize voice utilities.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
    
    def normalize_voice_embedding(
        self,
        voice_embedding: np.ndarray,
        method: str = "minmax",
        target_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> np.ndarray:
        """
        Normalize voice embedding to a target range.
        
        Args:
            voice_embedding: Voice embedding to normalize
            method: Normalization method
            target_range: Target range for normalization
        
        Returns:
            Normalized voice embedding
        """
        if method == "minmax":
            min_val, max_val = np.min(voice_embedding), np.max(voice_embedding)
            if max_val - min_val == 0:
                return voice_embedding
            normalized = (voice_embedding - min_val) / (max_val - min_val)
            return normalized * (target_range[1] - target_range[0]) + target_range[0]
        
        elif method == "zscore":
            mean_val, std_val = np.mean(voice_embedding), np.std(voice_embedding)
            if std_val == 0:
                return voice_embedding
            normalized = (voice_embedding - mean_val) / std_val
            return np.clip(normalized, target_range[0], target_range[1])
        
        elif method == "l2":
            norm = np.linalg.norm(voice_embedding)
            if norm == 0:
                return voice_embedding
            return voice_embedding / norm
        
        else:
            return voice_embedding
    
    def denormalize_voice_embedding(
        self,
        normalized_embedding: np.ndarray,
        original_stats: Dict[str, float],
        method: str = "minmax",
    ) -> np.ndarray:
        """
        Denormalize voice embedding back to original scale.
        
        Args:
            normalized_embedding: Normalized voice embedding
            original_stats: Original statistics (min, max, mean, std)
            method: Denormalization method
        
        Returns:
            Denormalized voice embedding
        """
        if method == "minmax":
            min_val, max_val = original_stats['min'], original_stats['max']
            # Reverse minmax normalization
            denormalized = (normalized_embedding + 1.0) / 2.0  # [0, 1]
            return denormalized * (max_val - min_val) + min_val
        
        elif method == "zscore":
            mean_val, std_val = original_stats['mean'], original_stats['std']
            return normalized_embedding * std_val + mean_val
        
        else:
            return normalized_embedding
    
    def extract_voice_characteristics(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Extract voice characteristics from audio signal.
        
        Args:
            audio: Audio signal
            sample_rate: Audio sample rate
        
        Returns:
            Dictionary of voice characteristics
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        characteristics = {}
        
        # Basic audio statistics
        characteristics['rms_energy'] = float(np.sqrt(np.mean(audio ** 2)))
        characteristics['peak_amplitude'] = float(np.max(np.abs(audio)))
        characteristics['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(audio)[0]))
        
        # Spectral characteristics
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        characteristics['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        characteristics['spectral_centroid_std'] = float(np.std(spectral_centroid))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
        characteristics['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        characteristics['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
        
        # Pitch characteristics
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            characteristics['pitch_mean'] = float(np.mean(pitch_values))
            characteristics['pitch_std'] = float(np.std(pitch_values))
            characteristics['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
        else:
            characteristics['pitch_mean'] = 0.0
            characteristics['pitch_std'] = 0.0
            characteristics['pitch_range'] = 0.0
        
        # MFCC characteristics
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        characteristics['mfcc_mean'] = float(np.mean(mfcc))
        characteristics['mfcc_std'] = float(np.std(mfcc))
        
        # Spectral features
        characteristics['spectral_bandwidth'] = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]))
        characteristics['spectral_contrast'] = float(np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0]))
        
        return characteristics
    
    def create_voice_embedding_from_audio(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        embedding_dim: int = 64,
    ) -> np.ndarray:
        """
        Create voice embedding from audio signal.
        
        Args:
            audio: Audio signal
            sample_rate: Audio sample rate
            embedding_dim: Embedding dimension
        
        Returns:
            Voice embedding vector
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Extract characteristics
        characteristics = self.extract_voice_characteristics(audio, sample_rate)
        
        # Create embedding from characteristics
        embedding = np.zeros(embedding_dim)
        
        # Map characteristics to embedding dimensions
        if embedding_dim >= 8:
            embedding[0] = characteristics['pitch_mean'] / 300.0  # Normalize pitch
            embedding[1] = characteristics['pitch_std'] / 100.0   # Normalize pitch std
            embedding[2] = characteristics['spectral_centroid_mean'] / 4000.0  # Normalize spectral centroid
            embedding[3] = characteristics['spectral_rolloff_mean'] / 8000.0   # Normalize spectral rolloff
            embedding[4] = characteristics['zero_crossing_rate']               # Zero crossing rate
            embedding[5] = len(audio) / (sample_rate * 10.0)                 # Duration factor
            embedding[6] = characteristics['mfcc_mean']                       # MFCC mean
            embedding[7] = characteristics['mfcc_std']                        # MFCC std
        
        # Fill remaining dimensions with derived features
        for i in range(8, embedding_dim):
            if i < len(characteristics):
                # Use additional characteristics if available
                char_values = list(characteristics.values())
                embedding[i] = char_values[i % len(char_values)]
            else:
                # Fill with random values based on existing characteristics
                embedding[i] = np.random.normal(0, 0.1)
        
        return embedding
    
    def convert_voice_characteristics(
        self,
        source_embedding: np.ndarray,
        target_characteristics: Dict[str, float],
        conversion_strength: float = 1.0,
    ) -> np.ndarray:
        """
        Convert voice characteristics while preserving voice identity.
        
        Args:
            source_embedding: Source voice embedding
            target_characteristics: Target characteristics to apply
            conversion_strength: Strength of conversion (0-1)
        
        Returns:
            Converted voice embedding
        """
        converted_embedding = source_embedding.copy()
        
        # Apply characteristic conversions
        for char_name, target_value in target_characteristics.items():
            if char_name == 'pitch' and len(converted_embedding) > 0:
                # Convert pitch
                current_pitch = converted_embedding[0] * 300  # Denormalize
                target_pitch = target_value
                new_pitch = current_pitch + (target_pitch - current_pitch) * conversion_strength
                converted_embedding[0] = new_pitch / 300.0  # Renormalize
            
            elif char_name == 'energy' and len(converted_embedding) > 2:
                # Convert energy (spectral centroid)
                current_energy = converted_embedding[2] * 4000  # Denormalize
                target_energy = target_value
                new_energy = current_energy + (target_energy - current_energy) * conversion_strength
                converted_embedding[2] = new_energy / 4000.0  # Renormalize
            
            elif char_name == 'brightness' and len(converted_embedding) > 3:
                # Convert brightness (spectral rolloff)
                current_brightness = converted_embedding[3] * 8000  # Denormalize
                target_brightness = target_value
                new_brightness = current_brightness + (target_brightness - current_brightness) * conversion_strength
                converted_embedding[3] = new_brightness / 8000.0  # Renormalize
        
        return converted_embedding
    
    def blend_voice_characteristics(
        self,
        voice1_embedding: np.ndarray,
        voice2_embedding: np.ndarray,
        blend_weights: Dict[str, float],
    ) -> np.ndarray:
        """
        Blend specific characteristics between two voices.
        
        Args:
            voice1_embedding: First voice embedding
            voice2_embedding: Second voice embedding
            blend_weights: Weights for blending each characteristic
        
        Returns:
            Blended voice embedding
        """
        blended_embedding = voice1_embedding.copy()
        
        # Blend characteristics based on weights
        for char_name, weight in blend_weights.items():
            if char_name == 'pitch' and len(blended_embedding) > 0:
                blended_embedding[0] = (1 - weight) * voice1_embedding[0] + weight * voice2_embedding[0]
            
            elif char_name == 'energy' and len(blended_embedding) > 2:
                blended_embedding[2] = (1 - weight) * voice1_embedding[2] + weight * voice2_embedding[2]
            
            elif char_name == 'brightness' and len(blended_embedding) > 3:
                blended_embedding[3] = (1 - weight) * voice1_embedding[3] + weight * voice2_embedding[3]
            
            elif char_name == 'warmth' and len(blended_embedding) > 4:
                blended_embedding[4] = (1 - weight) * voice1_embedding[4] + weight * voice2_embedding[4]
        
        return blended_embedding
    
    def resample_audio(
        self,
        audio: np.ndarray,
        original_sr: int,
        target_sr: int,
        method: str = "librosa",
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Audio signal
            original_sr: Original sample rate
            target_sr: Target sample rate
            method: Resampling method
        
        Returns:
            Resampled audio signal
        """
        if original_sr == target_sr:
            return audio
        
        if method == "librosa":
            return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
        
        elif method == "scipy":
            num_samples = int(len(audio) * target_sr / original_sr)
            return resample(audio, num_samples)
        
        else:
            return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    
    def apply_audio_effects(
        self,
        audio: np.ndarray,
        effects: Dict[str, float],
    ) -> np.ndarray:
        """
        Apply audio effects to audio signal.
        
        Args:
            audio: Audio signal
            effects: Dictionary of effects and their parameters
        
        Returns:
            Audio signal with effects applied
        """
        processed_audio = audio.copy()
        
        for effect_name, effect_value in effects.items():
            if effect_name == 'pitch_shift':
                # Apply pitch shifting
                processed_audio = librosa.effects.pitch_shift(
                    processed_audio, sr=self.sample_rate, n_steps=effect_value
                )
            
            elif effect_name == 'time_stretch':
                # Apply time stretching
                processed_audio = librosa.effects.time_stretch(
                    processed_audio, rate=effect_value
                )
            
            elif effect_name == 'volume':
                # Apply volume scaling
                processed_audio = processed_audio * effect_value
            
            elif effect_name == 'noise':
                # Add noise
                noise = np.random.normal(0, effect_value, len(processed_audio))
                processed_audio = processed_audio + noise
            
            elif effect_name == 'reverb':
                # Apply simple reverb (convolution with impulse response)
                # This is a simplified reverb implementation
                impulse_length = int(self.sample_rate * 0.5)  # 0.5 second impulse
                impulse = np.exp(-np.linspace(0, 10, impulse_length)) * effect_value
                processed_audio = np.convolve(processed_audio, impulse, mode='same')
        
        return processed_audio
    
    def create_voice_profile(
        self,
        voice_embedding: np.ndarray,
        audio: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Create a comprehensive voice profile.
        
        Args:
            voice_embedding: Voice embedding
            audio: Optional audio signal for additional analysis
        
        Returns:
            Voice profile dictionary
        """
        profile = {
            'embedding': voice_embedding.tolist(),
            'embedding_dim': len(voice_embedding),
            'characteristics': {},
            'quality_metrics': {},
        }
        
        # Extract characteristics from embedding
        if len(voice_embedding) >= 8:
            profile['characteristics'] = {
                'pitch_mean': float(voice_embedding[0] * 300),
                'pitch_variability': float(voice_embedding[1] * 100),
                'spectral_centroid': float(voice_embedding[2] * 4000),
                'spectral_rolloff': float(voice_embedding[3] * 8000),
                'zero_crossing_rate': float(voice_embedding[4]),
                'duration_factor': float(voice_embedding[5] * 10),
                'mfcc_mean': float(voice_embedding[6]),
                'mfcc_variability': float(voice_embedding[7]),
            }
        
        # Analyze audio if provided
        if audio is not None:
            audio_characteristics = self.extract_voice_characteristics(audio)
            profile['audio_characteristics'] = audio_characteristics
        
        # Quality metrics
        profile['quality_metrics'] = {
            'embedding_norm': float(np.linalg.norm(voice_embedding)),
            'embedding_std': float(np.std(voice_embedding)),
            'embedding_range': float(np.max(voice_embedding) - np.min(voice_embedding)),
        }
        
        return profile
    
    def save_voice_profile(
        self,
        voice_profile: Dict[str, Any],
        file_path: str,
    ) -> None:
        """
        Save voice profile to file.
        
        Args:
            voice_profile: Voice profile dictionary
            file_path: Path to save the profile
        """
        with open(file_path, 'w') as f:
            json.dump(voice_profile, f, indent=2)
        
        print(f"Voice profile saved to: {file_path}")
    
    def load_voice_profile(self, file_path: str) -> Dict[str, Any]:
        """
        Load voice profile from file.
        
        Args:
            file_path: Path to load the profile from
        
        Returns:
            Voice profile dictionary
        """
        with open(file_path, 'r') as f:
            voice_profile = json.load(f)
        
        # Convert embedding back to numpy array
        if 'embedding' in voice_profile:
            voice_profile['embedding'] = np.array(voice_profile['embedding'])
        
        return voice_profile
    
    def visualize_voice_characteristics(
        self,
        voice_embedding: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize voice characteristics.
        
        Args:
            voice_embedding: Voice embedding to visualize
            save_path: Path to save the visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Embedding values
        axes[0, 0].bar(range(len(voice_embedding)), voice_embedding)
        axes[0, 0].set_title('Voice Embedding Values')
        axes[0, 0].set_xlabel('Dimension')
        axes[0, 0].set_ylabel('Value')
        
        # 2. Embedding statistics
        stats = {
            'Mean': np.mean(voice_embedding),
            'Std': np.std(voice_embedding),
            'Min': np.min(voice_embedding),
            'Max': np.max(voice_embedding),
        }
        axes[0, 1].bar(stats.keys(), stats.values())
        axes[0, 1].set_title('Embedding Statistics')
        axes[0, 1].set_ylabel('Value')
        
        # 3. Voice characteristics (if available)
        if len(voice_embedding) >= 8:
            characteristics = {
                'Pitch': voice_embedding[0] * 300,
                'Pitch Var': voice_embedding[1] * 100,
                'Spectral Centroid': voice_embedding[2] * 4000,
                'Spectral Rolloff': voice_embedding[3] * 8000,
            }
            axes[1, 0].bar(characteristics.keys(), characteristics.values())
            axes[1, 0].set_title('Voice Characteristics')
            axes[1, 0].set_ylabel('Value')
        
        # 4. Embedding distribution
        axes[1, 1].hist(voice_embedding, bins=20, alpha=0.7)
        axes[1, 1].set_title('Embedding Distribution')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_voice_profiles(
        self,
        profile1: Dict[str, Any],
        profile2: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Compare two voice profiles.
        
        Args:
            profile1: First voice profile
            profile2: Second voice profile
        
        Returns:
            Comparison metrics
        """
        embedding1 = np.array(profile1['embedding'])
        embedding2 = np.array(profile2['embedding'])
        
        comparison = {
            'cosine_similarity': 1 - np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            ),
            'euclidean_distance': np.linalg.norm(embedding1 - embedding2),
            'manhattan_distance': np.sum(np.abs(embedding1 - embedding2)),
        }
        
        # Compare characteristics if available
        if 'characteristics' in profile1 and 'characteristics' in profile2:
            char1 = profile1['characteristics']
            char2 = profile2['characteristics']
            
            for char_name in char1:
                if char_name in char2:
                    comparison[f'{char_name}_difference'] = abs(char1[char_name] - char2[char_name])
        
        return comparison








































