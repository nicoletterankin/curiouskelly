#!/usr/bin/env python3
"""
Kelly25 Voice API Server
Production-ready API for Kelly25 voice synthesis
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import io
import json
import logging
from datetime import datetime
from pathlib import Path
import asyncio
from typing import List, Optional
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Kelly25 Voice API",
    description="AI-powered voice synthesis using Kelly25 model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SynthesisRequest(BaseModel):
    text: str
    output_format: str = "wav"  # wav, mp3
    sample_rate: int = 22050
    max_length: int = 50

class SynthesisResponse(BaseModel):
    request_id: str
    status: str
    audio_url: Optional[str] = None
    duration: Optional[float] = None
    text: str
    timestamp: str

class BatchSynthesisRequest(BaseModel):
    texts: List[str]
    output_format: str = "wav"
    sample_rate: int = 22050

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime: str

# Global model instance
class Kelly25Model:
    def __init__(self):
        self.model = None
        self.device = torch.device('cpu')
        self.load_model()
    
    def load_model(self):
        """Load the trained Kelly25 model"""
        try:
            # Model architecture (matching training)
            class BasicTTSModel(nn.Module):
                def __init__(self, vocab_size=256, hidden_dim=128, audio_length=110250):
                    super().__init__()
                    self.hidden_dim = hidden_dim
                    self.audio_length = audio_length
                    
                    # Text encoder
                    self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
                    self.text_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
                    self.text_proj = nn.Linear(hidden_dim * 2, hidden_dim)
                    
                    # Generator
                    self.generator = nn.Sequential(
                        nn.Linear(hidden_dim, 512),
                        nn.ReLU(),
                        nn.Linear(512, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, audio_length)
                    )
                
                def forward(self, text_ids):
                    embedded = self.text_embedding(text_ids)
                    encoded, _ = self.text_lstm(embedded)
                    projected = self.text_proj(encoded)
                    pooled = torch.mean(projected, dim=1)
                    audio = self.generator(pooled)
                    return audio
            
            # Initialize model
            self.model = BasicTTSModel(
                vocab_size=256,
                hidden_dim=128,
                audio_length=110250
            )
            
            # Load trained weights
            checkpoint = torch.load('kelly25_model_output/best_model.pth', map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            logger.info("Kelly25 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise e
    
    def text_to_ids(self, text, max_length=50):
        """Convert text to character IDs"""
        text_ids = [ord(c) for c in text[:max_length]]
        text_ids = text_ids + [0] * (max_length - len(text_ids))
        return torch.LongTensor(text_ids).unsqueeze(0)
    
    def synthesize(self, text, sample_rate=22050):
        """Synthesize audio from text"""
        with torch.no_grad():
            text_ids = self.text_to_ids(text)
            generated_audio = self.model(text_ids)
            audio_np = generated_audio.squeeze().numpy()
            return audio_np

# Initialize model
kelly25_model = Kelly25Model()

# In-memory storage for generated audio
audio_cache = {}
start_time = datetime.now()

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Kelly25 Voice API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "synthesize": "/synthesize",
            "batch_synthesize": "/batch-synthesize",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = datetime.now() - start_time
    return HealthResponse(
        status="healthy",
        model_loaded=kelly25_model.model is not None,
        version="1.0.0",
        uptime=str(uptime)
    )

@app.post("/synthesize", response_model=SynthesisResponse)
async def synthesize_speech(request: SynthesisRequest):
    """Synthesize speech from text"""
    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(request.text) > request.max_length:
            raise HTTPException(status_code=400, detail=f"Text too long. Maximum {request.max_length} characters")
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Synthesize audio
        logger.info(f"Synthesizing: '{request.text}'")
        audio = kelly25_model.synthesize(request.text, request.sample_rate)
        
        # Calculate duration
        duration = len(audio) / request.sample_rate
        
        # Save audio file
        output_dir = Path("api_output")
        output_dir.mkdir(exist_ok=True)
        
        if request.output_format.lower() == "wav":
            filename = f"{request_id}.wav"
            filepath = output_dir / filename
            sf.write(filepath, audio, request.sample_rate)
        else:
            # For MP3, we'd need additional conversion
            filename = f"{request_id}.wav"  # Fallback to WAV
            filepath = output_dir / filename
            sf.write(filepath, audio, request.sample_rate)
        
        # Store in cache
        audio_cache[request_id] = {
            "filepath": str(filepath),
            "text": request.text,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
        
        return SynthesisResponse(
            request_id=request_id,
            status="success",
            audio_url=f"/audio/{request_id}",
            duration=duration,
            text=request.text,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Synthesis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

@app.post("/batch-synthesize")
async def batch_synthesize(request: BatchSynthesisRequest):
    """Synthesize multiple texts in batch"""
    try:
        if len(request.texts) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 texts per batch")
        
        results = []
        for i, text in enumerate(request.texts):
            try:
                # Generate unique request ID
                request_id = str(uuid.uuid4())
                
                # Synthesize audio
                audio = kelly25_model.synthesize(text, request.sample_rate)
                duration = len(audio) / request.sample_rate
                
                # Save audio file
                output_dir = Path("api_output")
                output_dir.mkdir(exist_ok=True)
                
                filename = f"batch_{request_id}.wav"
                filepath = output_dir / filename
                sf.write(filepath, audio, request.sample_rate)
                
                # Store in cache
                audio_cache[request_id] = {
                    "filepath": str(filepath),
                    "text": text,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append({
                    "index": i,
                    "request_id": request_id,
                    "status": "success",
                    "audio_url": f"/audio/{request_id}",
                    "duration": duration,
                    "text": text
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "status": "error",
                    "error": str(e),
                    "text": text
                })
        
        return {
            "batch_id": str(uuid.uuid4()),
            "total_requests": len(request.texts),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"]),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch synthesis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch synthesis failed: {str(e)}")

@app.get("/audio/{request_id}")
async def get_audio(request_id: str):
    """Retrieve generated audio file"""
    if request_id not in audio_cache:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    filepath = audio_cache[request_id]["filepath"]
    if not Path(filepath).exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        filepath,
        media_type="audio/wav",
        filename=f"kelly25_{request_id}.wav"
    )

@app.get("/audio/{request_id}/stream")
async def stream_audio(request_id: str):
    """Stream audio file"""
    if request_id not in audio_cache:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    filepath = audio_cache[request_id]["filepath"]
    if not Path(filepath).exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    def iterfile():
        with open(filepath, mode="rb") as file_like:
            yield from file_like
    
    return StreamingResponse(
        iterfile(),
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename=kelly25_{request_id}.wav"}
    )

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    return {
        "total_cached": len(audio_cache),
        "cache_size_mb": sum(Path(info["filepath"]).stat().st_size for info in audio_cache.values()) / (1024 * 1024),
        "oldest_entry": min(info["timestamp"] for info in audio_cache.values()) if audio_cache else None,
        "newest_entry": max(info["timestamp"] for info in audio_cache.values()) if audio_cache else None
    }

@app.delete("/cache/clear")
async def clear_cache():
    """Clear audio cache"""
    global audio_cache
    cleared_count = len(audio_cache)
    audio_cache = {}
    return {"message": f"Cleared {cleared_count} cached audio files"}

@app.get("/examples")
async def get_examples():
    """Get example synthesis requests"""
    return {
        "examples": [
            {
                "text": "Hello! I'm Kelly, your learning companion.",
                "description": "Basic greeting"
            },
            {
                "text": "Let's explore this concept together.",
                "description": "Collaborative learning"
            },
            {
                "text": "Great job on that last attempt!",
                "description": "Encouragement"
            },
            {
                "text": "What do you think about this idea?",
                "description": "Question/engagement"
            },
            {
                "text": "Mathematics is the language of the universe.",
                "description": "Educational content"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





































