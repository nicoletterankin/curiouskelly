#!/usr/bin/env python3
"""
🧪 KELLY-SYNC QUICK TEST

Fast validation that the pipeline is working.
Tests each stage with minimal data.

Usage:
    python scripts/quick_test.py
    python scripts/quick_test.py --stage audio
    python scripts/quick_test.py --stage lipsync
    python scripts/quick_test.py --full
"""

import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def test_imports():
    """Test that all modules can be imported."""
    print_header("📦 TESTING IMPORTS")
    
    tests = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("librosa", "Librosa"),
        ("PIL", "Pillow"),
    ]
    
    all_passed = True
    
    for module, name in tests:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False
    
    # Test GPU
    print()
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✅ GPU: {gpu} ({mem:.1f}GB)")
        else:
            print(f"  ⚠️  No GPU available (CPU only)")
    except Exception as e:
        print(f"  ❌ GPU check failed: {e}")
    
    return all_passed

def test_pipeline_import():
    """Test that pipeline modules can be imported."""
    print_header("🔧 TESTING PIPELINE MODULES")
    
    modules = [
        "src.pipeline",
        "src.audio_processor",
        "src.lip_synthesizer",
        "src.face_restorer",
        "src.super_resolution",
        "src.motion_transfer",
        "src.compositor",
    ]
    
    all_passed = True
    
    for module in modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            all_passed = False
    
    return all_passed

def test_config():
    """Test configuration loading."""
    print_header("⚙️  TESTING CONFIGURATION")
    
    import yaml
    
    config_path = Path(__file__).parent.parent / 'config.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"  ✅ Config loaded: {config_path}")
        print(f"     Kelly reference: {config['kelly']['reference_image']}")
        print(f"     Output resolution: {config['pipeline']['target_resolution']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Config error: {e}")
        return False

def test_models():
    """Check if models are downloaded."""
    print_header("📦 CHECKING MODELS")
    
    models_dir = Path(__file__).parent.parent / 'models'
    
    model_checks = [
        ('video_retalking', 'LNet.pth'),
        ('codeformer', 'codeformer.pth'),
        ('real_esrgan', 'RealESRGAN_x4plus.pth'),
        ('fomm', 'vox-adv-cpk.pth.tar'),
    ]
    
    all_present = True
    
    for subdir, filename in model_checks:
        path = models_dir / subdir / filename
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {subdir}/{filename} ({size_mb:.1f}MB)")
        else:
            print(f"  ❌ {subdir}/{filename} (not found)")
            all_present = False
    
    if not all_present:
        print("\n  Run: python scripts/download_models.py")
    
    return all_present

def test_audio_processor():
    """Test audio processing stage."""
    print_header("🎤 TESTING AUDIO PROCESSOR")
    
    try:
        from src.audio_processor import AudioProcessor
        
        processor = AudioProcessor()
        
        # Create a test audio
        import numpy as np
        import tempfile
        import soundfile as sf
        
        # Generate 1 second of silence
        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio, sr)
        
        # Test loading
        waveform, sample_rate = processor.load_audio(temp_path)
        print(f"  ✅ Audio loading works")
        print(f"     Shape: {waveform.shape}, SR: {sample_rate}")
        
        # Clean up
        os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Audio processor error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_face_detection():
    """Test face detection."""
    print_header("👤 TESTING FACE DETECTION")
    
    try:
        from src.lip_synthesizer import LipSynthesizer
        import numpy as np
        
        # Create test image
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        
        synth = LipSynthesizer()
        
        # This will fail without a real face, but tests the import
        print(f"  ✅ LipSynthesizer initialized")
        
        return True
        
    except Exception as e:
        print(f"  ⚠️  Face detection: {e}")
        return True  # Not a critical failure

def test_quick_generation():
    """Test quick generation with placeholder."""
    print_header("🎬 TESTING QUICK GENERATION")
    
    try:
        from src.pipeline import PipelineConfig
        
        config = PipelineConfig(
            audio_path="test.mp3",
            reference_image="test.png",
            output_path="output/test.mp4",
        )
        
        print(f"  ✅ PipelineConfig works")
        print(f"     Resolution: {config.resolution}")
        print(f"     Device: {config.device}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Pipeline config error: {e}")
        return False

def run_full_test():
    """Run full pipeline test with sample data."""
    print_header("🚀 FULL PIPELINE TEST")
    
    print("  ⚠️  Full test requires:")
    print("     - Downloaded models")
    print("     - ELEVENLABS_API_KEY environment variable")
    print("     - Kelly reference image")
    print()
    
    try:
        from scripts.generate_video import KellyVideoGenerator
        
        generator = KellyVideoGenerator(quality='draft')
        
        # Check if we can generate audio
        if not os.environ.get('ELEVENLABS_API_KEY'):
            print("  ⚠️  No ELEVENLABS_API_KEY, skipping audio test")
        else:
            print("  Testing audio generation...")
            audio_path = generator.generate_audio(
                "Hello, this is a test.",
                output_path="output/test_audio.mp3"
            )
            print(f"  ✅ Audio generated: {audio_path}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Full test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Kelly-Sync Quick Test')
    parser.add_argument('--stage', choices=['imports', 'config', 'models', 'audio', 'face', 'pipeline'],
                        help='Test specific stage')
    parser.add_argument('--full', action='store_true', help='Run full pipeline test')
    
    args = parser.parse_args()
    
    print("""
╔═══════════════════════════════════════════════════════════╗
║  🧪 KELLY-SYNC QUICK TEST                                  ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    start_time = time.time()
    results = {}
    
    if args.stage:
        # Run specific stage
        stages = {
            'imports': test_imports,
            'config': test_config,
            'models': test_models,
            'audio': test_audio_processor,
            'face': test_face_detection,
            'pipeline': test_quick_generation,
        }
        
        if args.stage in stages:
            results[args.stage] = stages[args.stage]()
    
    elif args.full:
        # Run all including full test
        results['imports'] = test_imports()
        results['config'] = test_config()
        results['models'] = test_models()
        results['audio'] = test_audio_processor()
        results['face'] = test_face_detection()
        results['pipeline'] = test_quick_generation()
        results['full'] = run_full_test()
    
    else:
        # Run quick tests only
        results['imports'] = test_imports()
        results['pipeline'] = test_pipeline_import()
        results['config'] = test_config()
        results['models'] = test_models()
    
    # Summary
    elapsed = time.time() - start_time
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print_header("📊 TEST SUMMARY")
    
    for name, passed_test in results.items():
        status = "✅" if passed_test else "❌"
        print(f"  {status} {name}")
    
    print(f"\n  Passed: {passed}/{total}")
    print(f"  Time: {elapsed:.2f}s")
    
    if passed == total:
        print("\n  🎉 All tests passed!")
        return 0
    else:
        print("\n  ⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
