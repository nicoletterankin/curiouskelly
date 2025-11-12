"""
Inference engine for the TTS system.

This module provides high-level inference capabilities for generating
synthetic speech from text input.
"""

import torch
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import json
import os
from pathlib import Path

from ..models import FastPitch, HiFiGAN, SpeakerEmbedding
from ..data import TextProcessor
from .prosody_controller import ProsodyController
from .synthesizer import Synthesizer


class InferenceEngine:
    """
    High-level inference engine for TTS synthesis.
    
    This class provides a simplified interface for generating synthetic speech
    with various voice characteristics and emotional expressions.
    """
    
    def __init__(
        self,
        model_dir: str,
        config_path: str,
        device: str = "auto",
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_dir: Directory containing trained models
            config_path: Path to character voice configuration
            device: Device to run inference on
        """
        self.synthesizer = Synthesizer(model_dir, config_path, device)
        self.device = self.synthesizer.device
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def synthesize_text(
        self,
        text: str,
        emotion: str = "neutral",
        voice_archetype: Optional[str] = None,
        custom_voice_params: Optional[Dict] = None,
        output_path: Optional[str] = None,
        emphasis_markers: Optional[List[Tuple[int, int, str]]] = None,
    ) -> Union[torch.Tensor, str]:
        """
        Synthesize speech from text with specified characteristics.
        
        Args:
            text: Input text to synthesize
            emotion: Target emotion ('neutral', 'happy', 'sad', etc.)
            voice_archetype: Predefined voice archetype
            custom_voice_params: Custom voice parameters
            output_path: Optional path to save audio file
            emphasis_markers: Optional emphasis markers
        
        Returns:
            Generated audio tensor or file path
        """
        # Determine speaker embedding
        speaker_embedding = None
        
        if voice_archetype:
            from ..models.speaker_embedding import SyntheticVoiceGenerator
            voice_generator = SyntheticVoiceGenerator()
            speaker_embedding = voice_generator.create_voice_from_archetype(voice_archetype)
        elif custom_voice_params:
            from ..models.speaker_embedding import SyntheticVoiceGenerator
            voice_generator = SyntheticVoiceGenerator()
            speaker_embedding = voice_generator.create_custom_voice(**custom_voice_params)
        
        # Synthesize speech
        result = self.synthesizer.synthesize(
            text=text,
            emotion=emotion,
            speaker_embedding=speaker_embedding,
            emphasis=emphasis_markers,
            output_path=output_path,
        )
        
        return result
    
    def synthesize_with_emotion_sequence(
        self,
        text: str,
        emotion_sequence: List[Tuple[int, int, str]],
        voice_archetype: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Union[torch.Tensor, str]:
        """
        Synthesize speech with changing emotions throughout the text.
        
        Args:
            text: Input text to synthesize
            emotion_sequence: List of (start, end, emotion) tuples
            voice_archetype: Predefined voice archetype
            output_path: Optional path to save audio file
        
        Returns:
            Generated audio tensor or file path
        """
        # For now, use the dominant emotion
        # In a more sophisticated implementation, you'd segment the text
        # and synthesize each segment with its corresponding emotion
        emotions = [emotion for _, _, emotion in emotion_sequence]
        dominant_emotion = max(set(emotions), key=emotions.count)
        
        return self.synthesize_text(
            text=text,
            emotion=dominant_emotion,
            voice_archetype=voice_archetype,
            output_path=output_path,
        )
    
    def create_voice_demo(
        self,
        demo_texts: List[str],
        voice_archetypes: List[str],
        emotions: List[str],
        output_dir: str,
    ) -> None:
        """
        Create a voice demonstration with different archetypes and emotions.
        
        Args:
            demo_texts: List of texts to synthesize
            voice_archetypes: List of voice archetypes to demonstrate
            emotions: List of emotions to demonstrate
            output_dir: Directory to save demonstration files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Creating voice demonstration with {len(voice_archetypes)} archetypes and {len(emotions)} emotions...")
        
        for i, text in enumerate(demo_texts):
            for j, archetype in enumerate(voice_archetypes):
                for k, emotion in enumerate(emotions):
                    filename = f"demo_{i:02d}_{archetype}_{emotion}.wav"
                    output_path = os.path.join(output_dir, filename)
                    
                    self.synthesize_text(
                        text=text,
                        emotion=emotion,
                        voice_archetype=archetype,
                        output_path=output_path,
                    )
                    
                    print(f"Generated: {filename}")
        
        print(f"Voice demonstration complete. Files saved to: {output_dir}")
    
    def compare_voices(
        self,
        text: str,
        voice_archetypes: List[str],
        emotion: str = "neutral",
        output_dir: str = "voice_comparison",
    ) -> None:
        """
        Generate the same text with different voice archetypes for comparison.
        
        Args:
            text: Text to synthesize
            voice_archetypes: List of voice archetypes to compare
            emotion: Target emotion
            output_dir: Directory to save comparison files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating voice comparison for: '{text}'")
        
        for archetype in voice_archetypes:
            filename = f"comparison_{archetype}_{emotion}.wav"
            output_path = os.path.join(output_dir, filename)
            
            self.synthesize_text(
                text=text,
                emotion=emotion,
                voice_archetype=archetype,
                output_path=output_path,
            )
            
            print(f"Generated: {filename}")
        
        print(f"Voice comparison complete. Files saved to: {output_dir}")
    
    def generate_emotional_speech(
        self,
        text: str,
        voice_archetype: str = "young_female",
        output_dir: str = "emotional_speech",
    ) -> None:
        """
        Generate the same text with different emotions.
        
        Args:
            text: Text to synthesize
            voice_archetype: Voice archetype to use
            output_dir: Directory to save emotional speech files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        emotions = self.synthesizer.get_available_emotions()
        
        print(f"Generating emotional speech for: '{text}'")
        
        for emotion in emotions:
            filename = f"emotional_{emotion}_{voice_archetype}.wav"
            output_path = os.path.join(output_dir, filename)
            
            self.synthesize_text(
                text=text,
                emotion=emotion,
                voice_archetype=voice_archetype,
                output_path=output_path,
            )
            
            print(f"Generated: {filename}")
        
        print(f"Emotional speech generation complete. Files saved to: {output_dir}")
    
    def create_voice_variants(
        self,
        text: str,
        n_variants: int = 5,
        emotion: str = "neutral",
        output_dir: str = "voice_variants",
    ) -> None:
        """
        Create multiple variants of the same voice.
        
        Args:
            text: Text to synthesize
            n_variants: Number of variants to create
            emotion: Target emotion
            output_dir: Directory to save variant files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Creating {n_variants} voice variants for: '{text}'")
        
        variants = self.synthesizer.create_voice_variants(
            text=text,
            n_variants=n_variants,
            emotion=emotion,
            output_dir=output_dir,
        )
        
        print(f"Voice variants complete. Files saved to: {output_dir}")
    
    def batch_synthesize(
        self,
        texts: List[str],
        voice_archetype: str = "young_female",
        emotion: str = "neutral",
        output_dir: str = "batch_synthesis",
    ) -> None:
        """
        Synthesize multiple texts in batch.
        
        Args:
            texts: List of texts to synthesize
            voice_archetype: Voice archetype to use
            emotion: Target emotion
            output_dir: Directory to save batch files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Batch synthesizing {len(texts)} texts...")
        
        for i, text in enumerate(texts):
            filename = f"batch_{i:03d}_{emotion}.wav"
            output_path = os.path.join(output_dir, filename)
            
            self.synthesize_text(
                text=text,
                emotion=emotion,
                voice_archetype=voice_archetype,
                output_path=output_path,
            )
            
            print(f"Generated: {filename}")
        
        print(f"Batch synthesis complete. Files saved to: {output_dir}")
    
    def get_voice_info(self) -> Dict:
        """Get information about available voices and capabilities."""
        return {
            'available_emotions': self.synthesizer.get_available_emotions(),
            'voice_characteristics': self.synthesizer.get_voice_characteristics(),
            'synthesis_info': self.synthesizer.get_synthesis_info(),
        }
    
    def test_synthesis_quality(
        self,
        test_texts: List[str],
        output_dir: str = "quality_test",
    ) -> Dict:
        """
        Test synthesis quality with various texts and voices.
        
        Args:
            test_texts: List of test texts
            output_dir: Directory to save test files
        
        Returns:
            Dictionary with quality test results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'test_texts': test_texts,
            'generated_files': [],
            'errors': [],
        }
        
        print("Running synthesis quality test...")
        
        for i, text in enumerate(test_texts):
            try:
                filename = f"quality_test_{i:03d}.wav"
                output_path = os.path.join(output_dir, filename)
                
                self.synthesize_text(
                    text=text,
                    emotion="neutral",
                    output_path=output_path,
                )
                
                results['generated_files'].append(output_path)
                print(f"✓ Generated: {filename}")
                
            except Exception as e:
                error_msg = f"Error synthesizing text {i}: {str(e)}"
                results['errors'].append(error_msg)
                print(f"✗ {error_msg}")
        
        print(f"Quality test complete. Generated {len(results['generated_files'])} files.")
        if results['errors']:
            print(f"Encountered {len(results['errors'])} errors.")
        
        return results








































