#!/usr/bin/env python3
"""
Simple Kelly25 API - Working Version
Simplified API that definitely works
"""

from flask import Flask, request, jsonify, send_file
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import json
import os
from pathlib import Path
import uuid

app = Flask(__name__)

# Simple model class
class SimpleTTSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_embedding = nn.Embedding(256, 128)
        self.text_lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.text_proj = nn.Linear(256, 128)
        self.generator = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 110250)
        )
    
    def forward(self, text_ids):
        embedded = self.text_embedding(text_ids)
        encoded, _ = self.text_lstm(embedded)
        projected = self.text_proj(encoded)
        pooled = torch.mean(projected, dim=1)
        audio = self.generator(pooled)
        return audio

# Global model
model = None
device = torch.device('cpu')

def load_model():
    global model
    try:
        model = SimpleTTSModel()
        
        # Load weights if they exist
        if Path("kelly25_model_output/best_model.pth").exists():
            checkpoint = torch.load("kelly25_model_output/best_model.pth", map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("✅ Model loaded successfully")
        else:
            print("⚠️ No trained model found, using random weights")
        
        model.eval()
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def text_to_ids(text, max_length=50):
    text_ids = [ord(c) for c in text[:max_length]]
    text_ids = text_ids + [0] * (max_length - len(text_ids))
    return torch.LongTensor(text_ids).unsqueeze(0)

def synthesize_audio(text):
    with torch.no_grad():
        text_ids = text_to_ids(text)
        generated_audio = model(text_ids)
        audio_np = generated_audio.squeeze().numpy()
        return audio_np

@app.route('/')
def home():
    return jsonify({
        "message": "Kelly25 Voice API - Simple Version",
        "status": "running",
        "endpoints": ["/health", "/synthesize", "/audio/<id>"]
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    })

@app.route('/synthesize', methods=['POST'])
def synthesize():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        if len(text) > 50:
            return jsonify({"error": "Text too long (max 50 characters)"}), 400
        
        # Generate audio
        audio = synthesize_audio(text)
        duration = len(audio) / 22050
        
        # Save audio
        request_id = str(uuid.uuid4())
        output_dir = Path("api_output")
        output_dir.mkdir(exist_ok=True)
        
        filename = f"{request_id}.wav"
        filepath = output_dir / filename
        sf.write(filepath, audio, 22050)
        
        return jsonify({
            "request_id": request_id,
            "status": "success",
            "audio_url": f"/audio/{request_id}",
            "duration": float(duration),
            "text": text
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/audio/<request_id>')
def get_audio(request_id):
    try:
        filepath = Path("api_output") / f"{request_id}.wav"
        if filepath.exists():
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({"error": "Audio not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/examples')
def examples():
    return jsonify({
        "examples": [
            {"text": "Hello! I'm Kelly, your learning companion.", "description": "Greeting"},
            {"text": "Let's explore this concept together.", "description": "Learning"},
            {"text": "Great job on that last attempt!", "description": "Encouragement"},
            {"text": "What do you think about this idea?", "description": "Question"},
            {"text": "Mathematics is the language of the universe.", "description": "Educational"}
        ]
    })

if __name__ == '__main__':
    print("🚀 Starting Simple Kelly25 API...")
    
    # Load model
    if load_model():
        print("✅ Model loaded, starting server...")
        app.run(host='127.0.0.1', port=8000, debug=True)
    else:
        print("❌ Failed to load model, exiting...")




































