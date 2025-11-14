#!/usr/bin/env python3
"""
Kelly25 Voice API Client
Python client for interacting with the Kelly25 Voice API
"""

import requests
import json
import time
from pathlib import Path
from typing import List, Optional

class Kelly25Client:
    """Client for Kelly25 Voice API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> dict:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def synthesize(self, text: str, output_format: str = "wav", sample_rate: int = 22050) -> dict:
        """Synthesize speech from text"""
        payload = {
            "text": text,
            "output_format": output_format,
            "sample_rate": sample_rate
        }
        
        response = self.session.post(f"{self.base_url}/synthesize", json=payload)
        response.raise_for_status()
        return response.json()
    
    def batch_synthesize(self, texts: List[str], output_format: str = "wav", sample_rate: int = 22050) -> dict:
        """Synthesize multiple texts"""
        payload = {
            "texts": texts,
            "output_format": output_format,
            "sample_rate": sample_rate
        }
        
        response = self.session.post(f"{self.base_url}/batch-synthesize", json=payload)
        response.raise_for_status()
        return response.json()
    
    def download_audio(self, request_id: str, output_path: str) -> bool:
        """Download generated audio file"""
        try:
            response = self.session.get(f"{self.base_url}/audio/{request_id}")
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    
    def get_examples(self) -> dict:
        """Get example synthesis requests"""
        response = self.session.get(f"{self.base_url}/examples")
        response.raise_for_status()
        return response.json()
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        response = self.session.get(f"{self.base_url}/cache/stats")
        response.raise_for_status()
        return response.json()

def demo_usage():
    """Demonstrate Kelly25 API usage"""
    print("🎤 Kelly25 Voice API Client Demo")
    print("=" * 40)
    
    # Initialize client
    client = Kelly25Client()
    
    try:
        # Health check
        print("1. Checking API health...")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Model loaded: {health['model_loaded']}")
        print()
        
        # Single synthesis
        print("2. Synthesizing single text...")
        text = "Hello! I'm Kelly, your learning companion."
        result = client.synthesize(text)
        print(f"   Text: '{result['text']}'")
        print(f"   Request ID: {result['request_id']}")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"   Audio URL: {result['audio_url']}")
        print()
        
        # Download audio
        print("3. Downloading audio...")
        output_path = f"demo_output_{result['request_id']}.wav"
        if client.download_audio(result['request_id'], output_path):
            print(f"   Audio saved to: {output_path}")
        print()
        
        # Batch synthesis
        print("4. Batch synthesis...")
        texts = [
            "Let's explore this concept together.",
            "Great job on that last attempt!",
            "What do you think about this idea?"
        ]
        batch_result = client.batch_synthesize(texts)
        print(f"   Total requests: {batch_result['total_requests']}")
        print(f"   Successful: {batch_result['successful']}")
        print(f"   Failed: {batch_result['failed']}")
        print()
        
        # Download batch audio
        print("5. Downloading batch audio...")
        for i, result in enumerate(batch_result['results']):
            if result['status'] == 'success':
                output_path = f"batch_demo_{i}_{result['request_id']}.wav"
                if client.download_audio(result['request_id'], output_path):
                    print(f"   Batch audio {i+1} saved to: {output_path}")
        print()
        
        # Cache stats
        print("6. Cache statistics...")
        cache_stats = client.get_cache_stats()
        print(f"   Cached files: {cache_stats['total_cached']}")
        print(f"   Cache size: {cache_stats['cache_size_mb']:.2f} MB")
        print()
        
        print("✅ Demo completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server.")
        print("Please start the server first: python start_kelly25_api.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    demo_usage()





































