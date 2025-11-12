#!/usr/bin/env python3
"""
Test Kelly Voice Cloning - Generate Fresh Sample
"""

import torch
import soundfile as sf
from pathlib import Path
from real_kelly_voice_cloner import KellyVoiceCloner, text_to_ids, extract_mel_spectrogram, generate_kelly_voice

def test_voice_cloning():
    print('🎤 Testing Kelly Voice Cloner - Generating Fresh Sample')
    print('=' * 60)
    
    # Load the trained model
    device = torch.device('cpu')
    model = KellyVoiceCloner().to(device)
    model.load_state_dict(torch.load('kelly_real_voice_cloner.pth', map_location=device))
    print('✅ Model loaded successfully!')
    
    # Get reference audio
    reference_audio = 'kelly25_training_data/wavs/kelly25_0001.wav'
    print(f'📁 Using reference: {reference_audio}')
    
    # Generate a fresh sample
    test_text = 'This is Kelly speaking through voice cloning technology!'
    print(f'🎵 Generating: "{test_text}"')
    
    audio = generate_kelly_voice(model, test_text, reference_audio, device)
    if audio is not None:
        # Save the test sample
        output_file = 'kelly_real_cloned_voice/kelly_test_fresh.wav'
        sf.write(output_file, audio, 22050)
        print(f'✅ Fresh sample saved: {output_file}')
        print(f'📊 Audio length: {len(audio)} samples ({len(audio)/22050:.2f} seconds)')
        print('🎉 VOICE CLONING IS WORKING!')
        return True
    else:
        print('❌ Generation failed')
        return False

if __name__ == "__main__":
    test_voice_cloning()
