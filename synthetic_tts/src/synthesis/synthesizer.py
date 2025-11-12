"""
Main synthesizer class for TTS generation.
"""

import torch
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import json
import os
from pathlib import Path

from ..models import FastPitch, HiFiGAN, SpeakerEmbedding
from ..data import TextProcessor
from .prosody_controller import ProsodyController


class Synthesizer:
    """
    Main synthesizer class for generating synthetic speech.
    
    This class coordinates all components of the TTS pipeline to generate
    high-quality synthetic speech from text input.
    """
    
    def __init__(
        self,
        model_dir: str,
        config_path: str,
        device: str = "auto",
    ):
        """
        Initialize the synthesizer.
        
        Args:
            model_dir: Directory containing trained models
            config_path: Path to character voice configuration
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.model_dir = Path(model_dir)
        self.device = self._get_device(device)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.text_processor = TextProcessor()
        self.prosody_controller = ProsodyController()
        
        # Load models
        self._load_models()
        
        # Initialize synthesis parameters
        self.sample_rate = self.config.get('audio_settings', {}).get('sample_rate', 22050)
        self.hop_length = self.config.get('audio_settings', {}).get('hop_length', 256)
        self.win_length = self.config.get('audio_settings', {}).get('win_length', 1024)
        self.n_mels = self.config.get('audio_settings', {}).get('n_mels', 80)
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the appropriate device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_models(self):
        """Load all required models."""
        # Load FastPitch model
        self.fastpitch = FastPitch(
            n_mel_channels=self.n_mels,
            n_symbols=256,  # Phoneme vocabulary size
            speaker_embedding_dim=64,
        )
        
        fastpitch_path = self.model_dir / "fastpitch.pt"
        if fastpitch_path.exists():
            checkpoint = torch.load(fastpitch_path, map_location=self.device)
            self.fastpitch.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Initialize with random weights if no trained model
            self.fastpitch.apply(self._init_weights)
        
        self.fastpitch.to(self.device)
        self.fastpitch.eval()
        
        # Load HiFi-GAN vocoder
        self.hifigan = HiFiGAN(
            n_mel_channels=self.n_mels,
            upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
        )
        
        hifigan_path = self.model_dir / "hifigan.pt"
        if hifigan_path.exists():
            checkpoint = torch.load(hifigan_path, map_location=self.device)
            self.hifigan.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Initialize with random weights if no trained model
            self.hifigan.apply(self._init_weights)
        
        self.hifigan.to(self.device)
        self.hifigan.eval()
        
        # Load speaker embedding
        self.speaker_embedding = SpeakerEmbedding(embedding_dim=64)
        speaker_path = self.model_dir / "speaker_embedding.pt"
        if speaker_path.exists():
            checkpoint = torch.load(speaker_path, map_location=self.device)
            self.speaker_embedding.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Initialize with random weights if no trained model
            self.speaker_embedding.apply(self._init_weights)
        
        self.speaker_embedding.to(self.device)
        self.speaker_embedding.eval()
    
    def _init_weights(self, m):
        """Initialize model weights."""
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def synthesize(
        self,
        text: str,
        emotion: str = "neutral",
        speaker_embedding: Optional[torch.Tensor] = None,
        emphasis: Optional[List[Tuple[int, int, str]]] = None,
        output_path: Optional[str] = None,
    ) -> Union[torch.Tensor, str]:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            emotion: Target emotion ('neutral', 'happy', 'sad', 'angry', etc.)
            speaker_embedding: Optional custom speaker embedding
            emphasis: Optional emphasis markers [(start, end, type), ...]
            output_path: Optional path to save audio file
        
        Returns:
            Generated audio tensor or path to saved file
        """
        # Process text
        text_metadata = self.text_processor.process_text_for_synthesis(
            text, emotion, emphasis
        )
        
        # Get speaker embedding
        if speaker_embedding is None:
            speaker_embedding = self._get_default_speaker_embedding()
        
        # Convert text to tokens
        tokens = torch.tensor(text_metadata['tokens'], dtype=torch.long).unsqueeze(0)
        text_lengths = torch.tensor([len(text_metadata['tokens'])], dtype=torch.long)
        
        # Move to device
        tokens = tokens.to(self.device)
        text_lengths = text_lengths.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)
        
        # Generate mel-spectrogram
        with torch.no_grad():
            # Get base acoustic features
            outputs = self.fastpitch(
                tokens,
                text_lengths,
                speaker_embedding,
            )
            
            mel_outputs = outputs['mel_outputs']
            duration = outputs['duration']
            pitch = outputs['pitch']
            energy = outputs['energy']
            
            # Apply prosodic modifications
            modified_pitch, modified_duration, modified_energy = self.prosody_controller.apply_emotion(
                pitch, duration, energy, emotion, text_metadata
            )
            
            # Apply linguistic prosody
            modified_pitch, modified_duration, modified_energy = self.prosody_controller.apply_linguistic_prosody(
                modified_pitch, modified_duration, modified_energy, text_metadata
            )
            
            # Apply emphasis if provided
            if emphasis:
                modified_pitch, modified_duration, modified_energy = self.prosody_controller.apply_emphasis(
                    modified_pitch, modified_duration, modified_energy, emphasis
                )
            
            # Generate waveform
            audio = self.hifigan(mel_outputs)
            audio = audio.squeeze(1)  # Remove channel dimension
        
        # Convert to numpy and normalize
        audio_np = audio.cpu().numpy()
        audio_np = self._normalize_audio(audio_np)
        
        # Save to file if path provided
        if output_path:
            self._save_audio(audio_np, output_path)
            return output_path
        else:
            return torch.tensor(audio_np, dtype=torch.float32)
    
    def _get_default_speaker_embedding(self) -> torch.Tensor:
        """Get default speaker embedding from configuration."""
        if 'speaker_embedding' in self.config:
            embedding = torch.tensor(
                self.config['speaker_embedding'],
                dtype=torch.float32
            ).unsqueeze(0)
        else:
            # Generate random embedding
            embedding = torch.randn(1, 64, dtype=torch.float32)
        
        return embedding
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping."""
        # Normalize to [-1, 1] range
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95  # Leave some headroom
        
        return audio
    
    def _save_audio(self, audio: np.ndarray, output_path: str):
        """Save audio to file."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to tensor and save
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        torchaudio.save(
            output_path,
            audio_tensor.unsqueeze(0),  # Add channel dimension
            self.sample_rate,
        )
    
    def synthesize_batch(
        self,
        texts: List[str],
        emotions: Optional[List[str]] = None,
        speaker_embeddings: Optional[List[torch.Tensor]] = None,
        output_dir: Optional[str] = None,
    ) -> List[Union[torch.Tensor, str]]:
        """
        Synthesize multiple texts in batch.
        
        Args:
            texts: List of texts to synthesize
            emotions: Optional list of emotions for each text
            speaker_embeddings: Optional list of speaker embeddings
            output_dir: Optional directory to save audio files
        
        Returns:
            List of generated audio tensors or file paths
        """
        if emotions is None:
            emotions = ['neutral'] * len(texts)
        
        if speaker_embeddings is None:
            speaker_embeddings = [self._get_default_speaker_embedding()] * len(texts)
        
        results = []
        for i, (text, emotion, speaker_embedding) in enumerate(zip(texts, emotions, speaker_embeddings)):
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, f"synthesis_{i:03d}.wav")
            
            result = self.synthesize(
                text=text,
                emotion=emotion,
                speaker_embedding=speaker_embedding,
                output_path=output_path,
            )
            results.append(result)
        
        return results
    
    def create_voice_variants(
        self,
        text: str,
        n_variants: int = 5,
        emotion: str = "neutral",
        output_dir: Optional[str] = None,
    ) -> List[Union[torch.Tensor, str]]:
        """
        Create multiple voice variants of the same text.
        
        Args:
            text: Text to synthesize
            n_variants: Number of variants to create
            emotion: Target emotion
            output_dir: Optional directory to save audio files
        
        Returns:
            List of generated audio variants
        """
        # Generate different speaker embeddings
        base_embedding = self._get_default_speaker_embedding()
        variants = []
        
        for i in range(n_variants):
            # Create variant embedding with slight modifications
            variation = torch.randn_like(base_embedding) * 0.1
            variant_embedding = base_embedding + variation
            
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, f"variant_{i:03d}.wav")
            
            result = self.synthesize(
                text=text,
                emotion=emotion,
                speaker_embedding=variant_embedding,
                output_path=output_path,
            )
            variants.append(result)
        
        return variants
    
    def get_voice_characteristics(self) -> Dict:
        """Get current voice characteristics from configuration."""
        return self.config.get('voice_parameters', {})
    
    def update_voice_characteristics(self, new_params: Dict):
        """Update voice characteristics."""
        if 'voice_parameters' not in self.config:
            self.config['voice_parameters'] = {}
        
        self.config['voice_parameters'].update(new_params)
    
    def save_config(self, config_path: str):
        """Save current configuration to file."""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_available_emotions(self) -> List[str]:
        """Get list of available emotions."""
        return list(self.config.get('emotional_presets', {}).keys())
    
    def get_synthesis_info(self) -> Dict:
        """Get information about the synthesis setup."""
        return {
            'model_dir': str(self.model_dir),
            'device': str(self.device),
            'sample_rate': self.sample_rate,
            'n_mels': self.n_mels,
            'available_emotions': self.get_available_emotions(),
            'voice_characteristics': self.get_voice_characteristics(),
        }








































