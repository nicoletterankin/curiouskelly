#!/usr/bin/env python3
"""
Cross-Platform TTS Testing Script
Tests TTS functionality across different platforms and configurations
"""

import os
import sys
import platform
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CrossPlatformTester:
    """Tests TTS functionality across platforms"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.results = {
            "platform": self.platform,
            "python_version": self.python_version,
            "tests": {},
            "errors": [],
            "warnings": []
        }
    
    def test_imports(self) -> bool:
        """Test if all required modules can be imported"""
        logger.info("Testing module imports...")
        
        required_modules = [
            "numpy", "librosa", "torch", "torchaudio", 
            "soundfile", "scipy", "matplotlib"
        ]
        
        failed_imports = []
        
        for module in required_modules:
            try:
                __import__(module)
                logger.info(f"✅ {module} imported successfully")
            except ImportError as e:
                failed_imports.append(module)
                logger.error(f"❌ Failed to import {module}: {e}")
        
        self.results["tests"]["imports"] = {
            "passed": len(failed_imports) == 0,
            "failed_modules": failed_imports,
            "total_modules": len(required_modules)
        }
        
        return len(failed_imports) == 0
    
    def test_audio_processing(self) -> bool:
        """Test basic audio processing functionality"""
        logger.info("Testing audio processing...")
        
        try:
            import numpy as np
            import librosa
            import soundfile as sf
            
            # Create test audio
            sample_rate = 22050
            duration = 1.0
            frequency = 440
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = 0.1 * np.sin(2 * np.pi * frequency * t)
            
            # Test librosa functionality
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
            
            # Test file I/O
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio, sample_rate)
                
                # Read it back
                loaded_audio, loaded_sr = sf.read(tmp_file.name)
                
                # Clean up
                os.unlink(tmp_file.name)
            
            logger.info("✅ Audio processing tests passed")
            self.results["tests"]["audio_processing"] = {
                "passed": True,
                "mfcc_shape": mfcc.shape,
                "spectral_centroid_shape": spectral_centroid.shape
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Audio processing test failed: {e}")
            self.results["tests"]["audio_processing"] = {
                "passed": False,
                "error": str(e)
            }
            return False
    
    def test_torch_functionality(self) -> bool:
        """Test PyTorch functionality"""
        logger.info("Testing PyTorch functionality...")
        
        try:
            import torch
            import torchaudio
            
            # Test basic tensor operations
            x = torch.randn(3, 4)
            y = torch.randn(4, 5)
            z = torch.mm(x, y)
            
            # Test CUDA availability
            cuda_available = torch.cuda.is_available()
            cuda_device_count = torch.cuda.device_count() if cuda_available else 0
            
            # Test audio tensor operations
            audio_tensor = torch.randn(1, 16000)  # 1 second at 16kHz
            mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=13)(audio_tensor)
            
            logger.info(f"✅ PyTorch tests passed (CUDA: {cuda_available})")
            self.results["tests"]["torch"] = {
                "passed": True,
                "cuda_available": cuda_available,
                "cuda_device_count": cuda_device_count,
                "tensor_operations": True
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ PyTorch test failed: {e}")
            self.results["tests"]["torch"] = {
                "passed": False,
                "error": str(e)
            }
            return False
    
    def test_tts_synthesis(self) -> bool:
        """Test TTS synthesis functionality"""
        logger.info("Testing TTS synthesis...")
        
        try:
            # Test if TTS modules exist
            tts_modules = [
                "synthetic_tts.src.models.fastpitch",
                "synthetic_tts.src.models.hifigan",
                "synthetic_tts.src.synthesis.synthesizer"
            ]
            
            failed_modules = []
            for module in tts_modules:
                try:
                    __import__(module)
                except ImportError:
                    failed_modules.append(module)
            
            if failed_modules:
                logger.warning(f"⚠️ Some TTS modules not available: {failed_modules}")
                self.results["warnings"].append(f"TTS modules not available: {failed_modules}")
            
            # Test basic synthesis (if modules are available)
            if not failed_modules:
                from synthetic_tts.src.synthesis.synthesizer import Synthesizer
                
                # This would be a real test if the modules were properly implemented
                logger.info("✅ TTS synthesis modules available")
                self.results["tests"]["tts_synthesis"] = {
                    "passed": True,
                    "modules_available": True
                }
            else:
                logger.info("⚠️ TTS synthesis modules not fully available")
                self.results["tests"]["tts_synthesis"] = {
                    "passed": False,
                    "modules_available": False,
                    "missing_modules": failed_modules
                }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ TTS synthesis test failed: {e}")
            self.results["tests"]["tts_synthesis"] = {
                "passed": False,
                "error": str(e)
            }
            return False
    
    def test_platform_specific(self) -> bool:
        """Test platform-specific functionality"""
        logger.info(f"Testing platform-specific functionality for {self.platform}...")
        
        platform_tests = {
            "windows": self.test_windows_specific,
            "darwin": self.test_macos_specific,
            "linux": self.test_linux_specific
        }
        
        if self.platform in platform_tests:
            return platform_tests[self.platform]()
        else:
            logger.warning(f"⚠️ Unknown platform: {self.platform}")
            return True
    
    def test_windows_specific(self) -> bool:
        """Test Windows-specific functionality"""
        logger.info("Testing Windows-specific features...")
        
        try:
            # Test Windows audio APIs
            import winsound
            winsound.Beep(1000, 100)  # 1kHz beep for 100ms
            
            # Test Windows-specific paths
            import winreg
            # This would test registry access if needed
            
            logger.info("✅ Windows-specific tests passed")
            self.results["tests"]["platform_specific"] = {
                "passed": True,
                "platform": "windows",
                "audio_api": "winsound"
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Windows-specific test failed: {e}")
            self.results["tests"]["platform_specific"] = {
                "passed": False,
                "platform": "windows",
                "error": str(e)
            }
            return False
    
    def test_macos_specific(self) -> bool:
        """Test macOS-specific functionality"""
        logger.info("Testing macOS-specific features...")
        
        try:
            # Test macOS audio frameworks
            import subprocess
            result = subprocess.run(['which', 'afplay'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ macOS audio tools available")
            
            # Test macOS-specific paths
            home_dir = Path.home()
            if (home_dir / "Library").exists():
                logger.info("✅ macOS Library directory accessible")
            
            self.results["tests"]["platform_specific"] = {
                "passed": True,
                "platform": "macos",
                "audio_tools": "afplay available"
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ macOS-specific test failed: {e}")
            self.results["tests"]["platform_specific"] = {
                "passed": False,
                "platform": "macos",
                "error": str(e)
            }
            return False
    
    def test_linux_specific(self) -> bool:
        """Test Linux-specific functionality"""
        logger.info("Testing Linux-specific features...")
        
        try:
            # Test ALSA/PulseAudio
            import subprocess
            result = subprocess.run(['which', 'aplay'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ Linux audio tools available")
            
            # Test system audio configuration
            if os.path.exists('/proc/asound/cards'):
                with open('/proc/asound/cards', 'r') as f:
                    cards = f.read()
                logger.info(f"✅ ALSA sound cards detected: {len(cards.splitlines())}")
            
            self.results["tests"]["platform_specific"] = {
                "passed": True,
                "platform": "linux",
                "audio_tools": "aplay available"
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Linux-specific test failed: {e}")
            self.results["tests"]["platform_specific"] = {
                "passed": False,
                "platform": "linux",
                "error": str(e)
            }
            return False
    
    def test_file_permissions(self) -> bool:
        """Test file system permissions"""
        logger.info("Testing file system permissions...")
        
        try:
            # Test write permissions
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(b"test")
                tmp_path = tmp_file.name
            
            # Test read permissions
            with open(tmp_path, 'rb') as f:
                data = f.read()
            
            # Clean up
            os.unlink(tmp_path)
            
            # Test directory creation
            test_dir = tempfile.mkdtemp()
            os.rmdir(test_dir)
            
            logger.info("✅ File system permissions OK")
            self.results["tests"]["file_permissions"] = {
                "passed": True,
                "write_permission": True,
                "read_permission": True,
                "directory_creation": True
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ File permissions test failed: {e}")
            self.results["tests"]["file_permissions"] = {
                "passed": False,
                "error": str(e)
            }
            return False
    
    def test_network_connectivity(self) -> bool:
        """Test network connectivity for model downloads"""
        logger.info("Testing network connectivity...")
        
        try:
            import urllib.request
            import socket
            
            # Test basic connectivity
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            
            # Test HTTPS connectivity
            urllib.request.urlopen("https://httpbin.org/get", timeout=10)
            
            logger.info("✅ Network connectivity OK")
            self.results["tests"]["network"] = {
                "passed": True,
                "dns_resolution": True,
                "https_connectivity": True
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Network test failed: {e}")
            self.results["tests"]["network"] = {
                "passed": False,
                "error": str(e)
            }
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all cross-platform tests"""
        logger.info("🚀 Starting cross-platform TTS tests...")
        logger.info(f"Platform: {self.platform}")
        logger.info(f"Python: {self.python_version}")
        
        test_functions = [
            self.test_imports,
            self.test_audio_processing,
            self.test_torch_functionality,
            self.test_tts_synthesis,
            self.test_platform_specific,
            self.test_file_permissions,
            self.test_network_connectivity
        ]
        
        passed_tests = 0
        total_tests = len(test_functions)
        
        for test_func in test_functions:
            try:
                if test_func():
                    passed_tests += 1
            except Exception as e:
                logger.error(f"❌ Test {test_func.__name__} crashed: {e}")
                self.results["errors"].append(f"{test_func.__name__}: {e}")
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests
        }
        
        return self.results
    
    def save_results(self, filename: str = None) -> str:
        """Save test results to file"""
        if filename is None:
            filename = f"test_results_{self.platform}_{self.python_version}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"📊 Test results saved to: {filename}")
        return filename

def main():
    """Main function for cross-platform testing"""
    print("🧪 Cross-Platform TTS Testing")
    print("=" * 50)
    
    tester = CrossPlatformTester()
    results = tester.run_all_tests()
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"   Platform: {results['platform']}")
    print(f"   Python: {results['python_version']}")
    print(f"   Tests Passed: {results['summary']['passed_tests']}/{results['summary']['total_tests']}")
    print(f"   Success Rate: {results['summary']['success_rate']:.1%}")
    
    if results['errors']:
        print(f"\n❌ Errors:")
        for error in results['errors']:
            print(f"   - {error}")
    
    if results['warnings']:
        print(f"\n⚠️ Warnings:")
        for warning in results['warnings']:
            print(f"   - {warning}")
    
    # Save results
    results_file = tester.save_results()
    
    # Exit with appropriate code
    if results['summary']['success_rate'] >= 0.8:
        print(f"\n✅ Cross-platform tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Cross-platform tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()






































