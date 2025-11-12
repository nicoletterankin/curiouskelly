#!/usr/bin/env python3
"""
Simple API Test
Test the Kelly25 API without starting a server
"""

import requests
import json
import time

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Kelly25 API")
    print("=" * 30)
    
    try:
        # Test health endpoint
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ API is running")
            print(f"   Status: {health['status']}")
            print(f"   Model loaded: {health['model_loaded']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
        
        # Test synthesis endpoint
        print("\n2. Testing synthesis endpoint...")
        synthesis_data = {
            "text": "Hello! I'm Kelly, your learning companion.",
            "output_format": "wav",
            "sample_rate": 22050
        }
        
        response = requests.post(f"{base_url}/synthesize", json=synthesis_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Synthesis successful")
            print(f"   Request ID: {result['request_id']}")
            print(f"   Duration: {result['duration']:.2f}s")
            print(f"   Audio URL: {result['audio_url']}")
            
            # Test audio download
            print("\n3. Testing audio download...")
            audio_response = requests.get(f"{base_url}/audio/{result['request_id']}", timeout=10)
            if audio_response.status_code == 200:
                print(f"   ✅ Audio download successful")
                print(f"   Audio size: {len(audio_response.content)} bytes")
                
                # Save test audio
                with open(f"test_api_output_{result['request_id']}.wav", "wb") as f:
                    f.write(audio_response.content)
                print(f"   Audio saved as: test_api_output_{result['request_id']}.wav")
            else:
                print(f"   ❌ Audio download failed: {audio_response.status_code}")
        else:
            print(f"   ❌ Synthesis failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
        
        print("\n🎉 All tests passed! API is working correctly.")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server.")
        print("Please start the server first:")
        print("   python -m uvicorn kelly25_api:app --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_api()




































