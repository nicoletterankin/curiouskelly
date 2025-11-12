"""
Enhanced synthesizer with voice interpolation and advanced features.

This module provides an enhanced TTS synthesizer that supports voice interpolation,
real-time voice switching, and advanced prosodic control.
"""

import torch
import torchaudio
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import librosa
from pathlib import Path
import json
import time

from ..models.model_factory import ModelFactory
from ..voice.voice_interpolator import VoiceInterpolator
from ..voice.voice_analyzer import VoiceAnalyzer
from ..data.text_processor import TextProcessor
from ..data.hybrid_data_generator import HybridDataGenerator


class EnhancedSynthesizer:
    """
    Enhanced TTS synthesizer with voice interpolation and advanced features.
    
    This synthesizer supports:
    - Voice interpolation and morphing
    - Real-time voice switching
    - Advanced prosodic control
    - Multiple model architectures
    - Voice quality optimization
    """
    
    def __init__(
        self,
        config: Dict,
        model_factory: Optional[ModelFactory] = None,
        voice_interpolator: Optional[VoiceInterpolator] = None,
        voice_analyzer: Optional[VoiceAnalyzer] = None,
    ):
        """
        Initialize the enhanced synthesizer.
        
        Args:
            config: Configuration dictionary
            model_factory: Model factory for creating models
            voice_interpolator: Voice interpolation engine
            voice_analyzer: Voice analysis engine
        """
        self.config = config
        
        # Initialize components
        self.model_factory = model_factory or ModelFactory()
        self.voice_interpolator = voice_interpolator or VoiceInterpolator()
        self.voice_analyzer = voice_analyzer or VoiceAnalyzer()
        self.text_processor = TextProcessor()
        
        # Model components
        self.acoustic_model = None
        self.vocoder = None
        self.speaker_embedding = None
        self.voice_generator = None
        
        # Voice database
        self.voice_database = {}
        self.current_voice = None
        
        # Interpolation settings
        self.interpolation_enabled = True
        self.interpolation_method = "linear"
        
        # Real-time settings
        self.real_time_mode = False
        self.voice_cache = {}
        
        # Performance metrics
        self.performance_metrics = {
            'synthesis_times': [],
            'interpolation_times': [],
            'voice_switches': 0,
        }
    
    def initialize_models(
        self,
        acoustic_architecture: str = "fastpitch",
        vocoder_architecture: str = "hifigan",
        acoustic_kwargs: Optional[Dict] = None,
        vocoder_kwargs: Optional[Dict] = None,
    ) -> None:
        """
        Initialize TTS models.
        
        Args:
            acoustic_architecture: Acoustic model architecture
            vocoder_architecture: Vocoder architecture
            acoustic_kwargs: Acoustic model parameters
            vocoder_kwargs: Vocoder parameters
        """
        print(f"Initializing TTS models: {acoustic_architecture} + {vocoder_architecture}")
        
        # Create TTS system
        tts_system = self.model_factory.create_tts_system(
            acoustic_architecture=acoustic_architecture,
            vocoder_architecture=vocoder_architecture,
            acoustic_kwargs=acoustic_kwargs,
            vocoder_kwargs=vocoder_kwargs,
        )
        
        self.acoustic_model = tts_system['acoustic_model']
        self.vocoder = tts_system['vocoder']
        self.speaker_embedding = tts_system['speaker_embedding']
        self.voice_generator = tts_system['voice_generator']
        
        print("TTS models initialized successfully")
    
    def load_voice_database(self, voice_database: Dict[str, np.ndarray]) -> None:
        """
        Load a database of voice embeddings.
        
        Args:
            voice_database: Dictionary mapping voice IDs to embeddings
        """
        self.voice_database = voice_database
        print(f"Loaded voice database with {len(voice_database)} voices")
        
        # Analyze database quality
        if self.voice_analyzer:
            analysis = self.voice_analyzer.analyze_voice_database(voice_database)
            print(f"Database quality score: {analysis.get('quality_scores', {}).get('overall_quality', 0):.4f}")
    
    def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        voice_embedding: Optional[np.ndarray] = None,
        emotion: Optional[str] = None,
        prosody_control: Optional[Dict] = None,
        interpolation_target: Optional[str] = None,
        interpolation_weight: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synthesize speech with enhanced voice control.
        
        Args:
            text: Text to synthesize
            voice_id: ID of voice to use
            voice_embedding: Custom voice embedding
            emotion: Emotion to apply
            prosody_control: Prosodic control parameters
            interpolation_target: Target voice for interpolation
            interpolation_weight: Interpolation weight (0-1)
            **kwargs: Additional synthesis parameters
        
        Returns:
            Dictionary containing synthesis results
        """
        start_time = time.time()
        
        # Process text
        text_metadata = self.text_processor.process_text_for_synthesis(text)
        
        # Determine voice embedding
        if voice_embedding is not None:
            final_voice_embedding = voice_embedding
        elif voice_id is not None and voice_id in self.voice_database:
            final_voice_embedding = self.voice_database[voice_id]
        elif self.current_voice is not None:
            final_voice_embedding = self.current_voice
        else:
            # Generate default voice
            final_voice_embedding = self.voice_generator.generate_voice_embedding()
        
        # Apply voice interpolation if requested
        if interpolation_target and interpolation_target in self.voice_database:
            target_embedding = self.voice_database[interpolation_target]
            final_voice_embedding = self.voice_interpolator.interpolate_voices(
                final_voice_embedding,
                target_embedding,
                interpolation_weight,
                method=self.interpolation_method
            )
        
        # Apply emotion and prosody control
        if emotion or prosody_control:
            final_voice_embedding = self._apply_prosodic_control(
                final_voice_embedding, emotion, prosody_control
            )
        
        # Synthesize speech
        synthesis_result = self._synthesize_with_models(
            text, text_metadata, final_voice_embedding, **kwargs
        )
        
        # Update performance metrics
        synthesis_time = time.time() - start_time
        self.performance_metrics['synthesis_times'].append(synthesis_time)
        
        # Add metadata to result
        synthesis_result.update({
            'text': text,
            'voice_id': voice_id,
            'emotion': emotion,
            'interpolation_target': interpolation_target,
            'interpolation_weight': interpolation_weight,
            'synthesis_time': synthesis_time,
        })
        
        return synthesis_result
    
    def _synthesize_with_models(
        self,
        text: str,
        text_metadata: Dict,
        voice_embedding: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Synthesize speech using the loaded models."""
        if self.acoustic_model is None or self.vocoder is None:
            raise RuntimeError("Models not initialized. Call initialize_models() first.")
        
        # Convert to tensors
        text_tensor = torch.tensor(text_metadata['phoneme_ids'], dtype=torch.long).unsqueeze(0)
        text_lengths = torch.tensor([len(text_metadata['phoneme_ids'])], dtype=torch.long)
        voice_tensor = torch.tensor(voice_embedding, dtype=torch.float32).unsqueeze(0)
        
        # Generate mel-spectrogram
        with torch.no_grad():
            mel_output = self.acoustic_model(
                text_tensor, text_lengths, voice_tensor
            )
        
        # Generate audio
        with torch.no_grad():
            audio = self.vocoder(mel_output['mel_outputs'])
        
        # Convert to numpy
        audio_np = audio.squeeze(0).cpu().numpy()
        
        return {
            'audio': audio_np,
            'mel_spectrogram': mel_output['mel_outputs'].squeeze(0).cpu().numpy(),
            'duration': mel_output.get('duration', None),
            'attention_weights': mel_output.get('attention_weights', None),
        }
    
    def _apply_prosodic_control(
        self,
        voice_embedding: np.ndarray,
        emotion: Optional[str] = None,
        prosody_control: Optional[Dict] = None,
    ) -> np.ndarray:
        """Apply prosodic control to voice embedding."""
        modified_embedding = voice_embedding.copy()
        
        # Apply emotion
        if emotion:
            emotion_controls = self._get_emotion_controls(emotion)
            for i, (param, value) in enumerate(emotion_controls.items()):
                if i < len(modified_embedding):
                    modified_embedding[i] = value
        
        # Apply custom prosody control
        if prosody_control:
            for param, value in prosody_control.items():
                if param in ['pitch_shift', 'rate_scale', 'energy_scale']:
                    # Map prosody parameters to embedding dimensions
                    if param == 'pitch_shift' and len(modified_embedding) > 0:
                        modified_embedding[0] = np.clip(modified_embedding[0] + value, -1, 1)
                    elif param == 'rate_scale' and len(modified_embedding) > 5:
                        modified_embedding[5] = np.clip(modified_embedding[5] * value, -1, 1)
                    elif param == 'energy_scale' and len(modified_embedding) > 2:
                        modified_embedding[2] = np.clip(modified_embedding[2] * value, -1, 1)
        
        return modified_embedding
    
    def _get_emotion_controls(self, emotion: str) -> Dict[str, float]:
        """Get emotion-specific control parameters."""
        emotion_controls = {
            'happy': {'pitch_shift': 0.2, 'rate_scale': 1.1, 'energy_scale': 1.2},
            'sad': {'pitch_shift': -0.1, 'rate_scale': 0.9, 'energy_scale': 0.8},
            'angry': {'pitch_shift': 0.3, 'rate_scale': 1.2, 'energy_scale': 1.3},
            'excited': {'pitch_shift': 0.4, 'rate_scale': 1.3, 'energy_scale': 1.4},
            'calm': {'pitch_shift': -0.2, 'rate_scale': 0.8, 'energy_scale': 0.7},
            'neutral': {'pitch_shift': 0.0, 'rate_scale': 1.0, 'energy_scale': 1.0},
        }
        
        return emotion_controls.get(emotion, emotion_controls['neutral'])
    
    def create_voice_continuum(
        self,
        voice1_id: str,
        voice2_id: str,
        text: str,
        num_steps: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Create a voice continuum between two voices.
        
        Args:
            voice1_id: First voice ID
            voice2_id: Second voice ID
            text: Text to synthesize
            num_steps: Number of interpolation steps
            **kwargs: Additional synthesis parameters
        
        Returns:
            List of synthesis results for each step
        """
        if voice1_id not in self.voice_database or voice2_id not in self.voice_database:
            raise ValueError(f"Voice IDs not found: {voice1_id}, {voice2_id}")
        
        voice1_embedding = self.voice_database[voice1_id]
        voice2_embedding = self.voice_database[voice2_id]
        
        # Create voice continuum
        voice_continuum = self.voice_interpolator.create_voice_continuum(
            voice1_embedding, voice2_embedding, num_steps, self.interpolation_method
        )
        
        # Synthesize speech for each step
        continuum_results = []
        for i, voice_embedding in enumerate(voice_continuum):
            result = self.synthesize_speech(
                text=text,
                voice_embedding=voice_embedding,
                **kwargs
            )
            result['step'] = i
            result['weight'] = i / (num_steps - 1)
            continuum_results.append(result)
        
        return continuum_results
    
    def morph_voice(
        self,
        source_voices: List[str],
        target_voice_id: str,
        text: str,
        morphing_steps: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Morph voice from source voices to target voice.
        
        Args:
            source_voices: List of source voice IDs
            target_voice_id: Target voice ID
            text: Text to synthesize
            morphing_steps: Number of morphing steps
            **kwargs: Additional synthesis parameters
        
        Returns:
            List of synthesis results for each morphing step
        """
        # Validate voice IDs
        for voice_id in source_voices + [target_voice_id]:
            if voice_id not in self.voice_database:
                raise ValueError(f"Voice ID not found: {voice_id}")
        
        # Get embeddings
        source_embeddings = [self.voice_database[voice_id] for voice_id in source_voices]
        target_embedding = self.voice_database[target_voice_id]
        
        # Create voice morphing
        morphed_voices = self.voice_interpolator.create_voice_morphing(
            source_embeddings, target_embedding, morphing_steps, self.interpolation_method
        )
        
        # Synthesize speech for each step
        morphing_results = []
        for i, voice_embedding in enumerate(morphed_voices):
            result = self.synthesize_speech(
                text=text,
                voice_embedding=voice_embedding,
                **kwargs
            )
            result['step'] = i
            result['morphing_progress'] = i / (morphing_steps - 1)
            morphing_results.append(result)
        
        return morphing_results
    
    def switch_voice(self, voice_id: str) -> None:
        """
        Switch to a different voice.
        
        Args:
            voice_id: ID of voice to switch to
        """
        if voice_id not in self.voice_database:
            raise ValueError(f"Voice ID not found: {voice_id}")
        
        self.current_voice = self.voice_database[voice_id]
        self.performance_metrics['voice_switches'] += 1
        
        print(f"Switched to voice: {voice_id}")
    
    def set_interpolation_method(self, method: str) -> None:
        """
        Set the voice interpolation method.
        
        Args:
            method: Interpolation method name
        """
        if method not in self.voice_interpolator.interpolation_methods:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        self.interpolation_method = method
        print(f"Interpolation method set to: {method}")
    
    def enable_real_time_mode(self, enabled: bool = True) -> None:
        """
        Enable or disable real-time mode.
        
        Args:
            enabled: Whether to enable real-time mode
        """
        self.real_time_mode = enabled
        if enabled:
            print("Real-time mode enabled")
        else:
            print("Real-time mode disabled")
    
    def get_voice_similarity(
        self,
        voice1_id: str,
        voice2_id: str,
    ) -> float:
        """
        Get similarity between two voices.
        
        Args:
            voice1_id: First voice ID
            voice2_id: Second voice ID
        
        Returns:
            Similarity score (0-1)
        """
        if voice1_id not in self.voice_database or voice2_id not in self.voice_database:
            raise ValueError("Voice IDs not found in database")
        
        voice1_embedding = self.voice_database[voice1_id]
        voice2_embedding = self.voice_database[voice2_id]
        
        comparison = self.voice_analyzer.compare_voices(voice1_embedding, voice2_embedding)
        return comparison['cosine_similarity']
    
    def find_similar_voices(
        self,
        query_voice_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Find voices similar to a query voice.
        
        Args:
            query_voice_id: Query voice ID
            top_k: Number of similar voices to return
            similarity_threshold: Minimum similarity threshold
        
        Returns:
            List of (voice_id, similarity_score) tuples
        """
        if query_voice_id not in self.voice_database:
            raise ValueError(f"Voice ID not found: {query_voice_id}")
        
        query_embedding = self.voice_database[query_voice_id]
        
        return self.voice_analyzer.recommend_similar_voices(
            query_embedding, self.voice_database, top_k, similarity_threshold
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = self.performance_metrics.copy()
        
        if metrics['synthesis_times']:
            metrics['avg_synthesis_time'] = np.mean(metrics['synthesis_times'])
            metrics['min_synthesis_time'] = np.min(metrics['synthesis_times'])
            metrics['max_synthesis_time'] = np.max(metrics['synthesis_times'])
        
        if metrics['interpolation_times']:
            metrics['avg_interpolation_time'] = np.mean(metrics['interpolation_times'])
        
        return metrics
    
    def save_voice_database(self, file_path: str) -> None:
        """
        Save voice database to file.
        
        Args:
            file_path: Path to save the database
        """
        database_data = {
            'voice_database': {k: v.tolist() for k, v in self.voice_database.items()},
            'current_voice': self.current_voice.tolist() if self.current_voice is not None else None,
            'interpolation_method': self.interpolation_method,
            'real_time_mode': self.real_time_mode,
        }
        
        with open(file_path, 'w') as f:
            json.dump(database_data, f, indent=2)
        
        print(f"Voice database saved to: {file_path}")
    
    def load_voice_database(self, file_path: str) -> None:
        """
        Load voice database from file.
        
        Args:
            file_path: Path to load the database from
        """
        with open(file_path, 'r') as f:
            database_data = json.load(f)
        
        self.voice_database = {
            k: np.array(v) for k, v in database_data['voice_database'].items()
        }
        
        if database_data['current_voice'] is not None:
            self.current_voice = np.array(database_data['current_voice'])
        
        self.interpolation_method = database_data.get('interpolation_method', 'linear')
        self.real_time_mode = database_data.get('real_time_mode', False)
        
        print(f"Voice database loaded from: {file_path}")
    
    def create_voice_family(
        self,
        parent_voice_id: str,
        family_size: int = 5,
        variation_scale: float = 0.2,
    ) -> List[str]:
        """
        Create a family of related voices.
        
        Args:
            parent_voice_id: Parent voice ID
            family_size: Number of family members to create
            variation_scale: Scale of variation
        """
        if parent_voice_id not in self.voice_database:
            raise ValueError(f"Voice ID not found: {parent_voice_id}")
        
        parent_embedding = self.voice_database[parent_voice_id]
        
        # Create voice family
        family_voices = self.voice_interpolator.create_voice_family(
            parent_embedding, family_size, variation_scale
        )
        
        # Add to database
        family_ids = []
        for i, voice_embedding in enumerate(family_voices[1:], 1):  # Skip parent
            family_id = f"{parent_voice_id}_family_{i}"
            self.voice_database[family_id] = voice_embedding
            family_ids.append(family_id)
        
        print(f"Created voice family with {len(family_ids)} members")
        return family_ids








































