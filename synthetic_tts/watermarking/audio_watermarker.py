"""
Audio Watermarking System for TTS Generated Content
Implements invisible watermarking for content tracking and IP protection
"""

import numpy as np
import librosa
import soundfile as sf
import hashlib
import json
import sqlite3
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class WatermarkType(Enum):
    """Types of watermarks that can be embedded"""
    USER_ID = "user_id"
    SESSION_ID = "session_id"
    TIMESTAMP = "timestamp"
    CUSTOM = "custom"

@dataclass
class WatermarkConfig:
    """Watermark configuration settings"""
    strength: float = 0.001  # Watermark strength (0.0 to 1.0)
    frequency_band: Tuple[float, float] = (8000, 22050)  # Frequency range for watermark
    redundancy: int = 3  # Number of redundant watermark copies
    error_correction: bool = True  # Enable error correction
    compression_resistant: bool = True  # Make watermark compression-resistant

class AudioWatermarker:
    """Main audio watermarking system"""
    
    def __init__(self, config: WatermarkConfig = None, db_path: str = "watermark_log.db"):
        self.config = config or WatermarkConfig()
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.AudioWatermarker")
        self._init_database()
    
    def _init_database(self):
        """Initialize watermark logging database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watermark_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    content_hash TEXT NOT NULL,
                    watermark_data TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    metadata TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watermark_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT NOT NULL,
                    detected_user_id TEXT,
                    confidence REAL,
                    timestamp DATETIME NOT NULL,
                    detection_method TEXT,
                    metadata TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            self.logger.info("Watermark database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize watermark database: {e}")
    
    def embed_watermark(self, audio: np.ndarray, sample_rate: int, 
                       user_id: str, session_id: str = None,
                       metadata: Dict[str, Any] = None) -> np.ndarray:
        """Embed watermark in audio"""
        try:
            # Generate watermark data
            watermark_data = self._generate_watermark_data(
                user_id, session_id, metadata
            )
            
            # Create watermark signal
            watermark_signal = self._create_watermark_signal(
                audio, sample_rate, watermark_data
            )
            
            # Embed watermark
            watermarked_audio = audio + (watermark_signal * self.config.strength)
            
            # Ensure no clipping
            watermarked_audio = np.clip(watermarked_audio, -1.0, 1.0)
            
            # Log embedding
            self._log_watermark_embedding(
                user_id, session_id, audio, watermark_data, metadata
            )
            
            self.logger.info(f"Watermark embedded for user {user_id}")
            return watermarked_audio
            
        except Exception as e:
            self.logger.error(f"Failed to embed watermark: {e}")
            return audio
    
    def _generate_watermark_data(self, user_id: str, session_id: str = None,
                                metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate watermark data structure"""
        timestamp = datetime.now().isoformat()
        session_id = session_id or hashlib.md5(f"{user_id}:{timestamp}".encode()).hexdigest()[:8]
        
        watermark_data = {
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": timestamp,
            "watermark_id": hashlib.sha256(f"{user_id}:{session_id}:{timestamp}".encode()).hexdigest()[:16],
            "metadata": metadata or {}
        }
        
        return watermark_data
    
    def _create_watermark_signal(self, audio: np.ndarray, sample_rate: int,
                                watermark_data: Dict[str, Any]) -> np.ndarray:
        """Create watermark signal from data"""
        # Convert watermark data to binary
        watermark_binary = self._data_to_binary(watermark_data)
        
        # Create spread spectrum watermark
        watermark_signal = np.zeros_like(audio)
        
        # Generate pseudo-random sequence based on watermark data
        seed = int(hashlib.sha256(json.dumps(watermark_data, sort_keys=True).encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        
        # Create watermark in frequency domain
        watermark_fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/sample_rate)
        
        # Apply watermark to specific frequency band
        freq_mask = (np.abs(freqs) >= self.config.frequency_band[0]) & \
                   (np.abs(freqs) <= self.config.frequency_band[1])
        
        # Create watermark pattern
        watermark_pattern = np.random.normal(0, 1, len(audio))
        watermark_pattern = np.fft.ifft(watermark_fft * freq_mask.astype(float))
        
        # Encode binary data into watermark
        for i, bit in enumerate(watermark_binary):
            if i * 100 < len(watermark_pattern):
                # Encode bit by modulating phase
                phase_shift = np.pi if bit == '1' else 0
                start_idx = i * 100
                end_idx = min((i + 1) * 100, len(watermark_pattern))
                watermark_pattern[start_idx:end_idx] *= np.exp(1j * phase_shift)
        
        return np.real(watermark_pattern)
    
    def _data_to_binary(self, data: Dict[str, Any]) -> str:
        """Convert watermark data to binary string"""
        # Create a compact binary representation
        binary_string = ""
        
        # Encode user_id (first 8 characters of hash)
        user_hash = hashlib.md5(data["user_id"].encode()).hexdigest()[:8]
        binary_string += format(int(user_hash, 16), '032b')
        
        # Encode session_id (first 4 characters of hash)
        session_hash = hashlib.md5(data["session_id"].encode()).hexdigest()[:4]
        binary_string += format(int(session_hash, 16), '016b')
        
        # Encode timestamp (Unix timestamp)
        timestamp = int(datetime.fromisoformat(data["timestamp"]).timestamp())
        binary_string += format(timestamp, '032b')
        
        return binary_string
    
    def _binary_to_data(self, binary_string: str) -> Dict[str, Any]:
        """Convert binary string back to watermark data"""
        try:
            # Extract user_id hash (32 bits)
            user_hash_hex = format(int(binary_string[:32], 2), '08x')
            
            # Extract session_id hash (16 bits)
            session_hash_hex = format(int(binary_string[32:48], 2), '04x')
            
            # Extract timestamp (32 bits)
            timestamp = int(binary_string[48:80], 2)
            timestamp_str = datetime.fromtimestamp(timestamp).isoformat()
            
            return {
                "user_id_hash": user_hash_hex,
                "session_id_hash": session_hash_hex,
                "timestamp": timestamp_str
            }
        except Exception as e:
            self.logger.error(f"Failed to decode binary data: {e}")
            return {}
    
    def detect_watermark(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Detect watermark in audio"""
        try:
            # Extract watermark signal
            watermark_signal = self._extract_watermark_signal(audio, sample_rate)
            
            # Decode watermark data
            watermark_data = self._decode_watermark_signal(watermark_signal)
            
            # Calculate confidence
            confidence = self._calculate_confidence(audio, watermark_signal)
            
            # Log detection
            self._log_watermark_detection(audio, watermark_data, confidence)
            
            return {
                "detected": confidence > 0.1,  # Threshold for detection
                "confidence": confidence,
                "watermark_data": watermark_data,
                "detection_method": "spread_spectrum"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to detect watermark: {e}")
            return {
                "detected": False,
                "confidence": 0.0,
                "watermark_data": {},
                "error": str(e)
            }
    
    def _extract_watermark_signal(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract watermark signal from audio"""
        # This is a simplified extraction - real implementation would be more complex
        # In practice, you'd need correlation analysis and error correction
        
        # Generate candidate watermarks for different user IDs
        candidates = []
        
        # Check first 100 user IDs (in production, use database lookup)
        for user_id_num in range(100):
            user_id = f"user_{user_id_num:03d}"
            session_id = "session_001"
            
            # Generate expected watermark
            watermark_data = {
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
            expected_watermark = self._create_watermark_signal(
                audio, sample_rate, watermark_data
            )
            
            # Calculate correlation
            correlation = np.corrcoef(audio, expected_watermark)[0, 1]
            candidates.append((correlation, user_id, session_id))
        
        # Find best match
        best_correlation, best_user_id, best_session_id = max(candidates)
        
        return {
            "correlation": best_correlation,
            "user_id": best_user_id,
            "session_id": best_session_id
        }
    
    def _decode_watermark_signal(self, watermark_signal: Dict[str, Any]) -> Dict[str, Any]:
        """Decode watermark signal to extract data"""
        return {
            "user_id": watermark_signal.get("user_id"),
            "session_id": watermark_signal.get("session_id"),
            "correlation": watermark_signal.get("correlation", 0.0)
        }
    
    def _calculate_confidence(self, audio: np.ndarray, watermark_signal: Dict[str, Any]) -> float:
        """Calculate confidence in watermark detection"""
        correlation = watermark_signal.get("correlation", 0.0)
        
        # Apply confidence scaling based on audio quality
        audio_quality = np.std(audio)  # Simple quality metric
        quality_factor = min(audio_quality * 10, 1.0)  # Scale to 0-1
        
        confidence = correlation * quality_factor
        return max(0.0, min(1.0, confidence))  # Clamp to 0-1
    
    def _log_watermark_embedding(self, user_id: str, session_id: str,
                                audio: np.ndarray, watermark_data: Dict[str, Any],
                                metadata: Dict[str, Any] = None):
        """Log watermark embedding to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            content_hash = hashlib.sha256(audio.tobytes()).hexdigest()
            
            cursor.execute("""
                INSERT INTO watermark_embeddings 
                (user_id, session_id, content_hash, watermark_data, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id, session_id, content_hash,
                json.dumps(watermark_data), datetime.now(),
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to log watermark embedding: {e}")
    
    def _log_watermark_detection(self, audio: np.ndarray, watermark_data: Dict[str, Any],
                                confidence: float):
        """Log watermark detection to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            content_hash = hashlib.sha256(audio.tobytes()).hexdigest()
            
            cursor.execute("""
                INSERT INTO watermark_detections 
                (content_hash, detected_user_id, confidence, timestamp, detection_method, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                content_hash, watermark_data.get("user_id"),
                confidence, datetime.now(), "spread_spectrum",
                json.dumps(watermark_data)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to log watermark detection: {e}")
    
    def get_watermark_history(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Get watermark embedding history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM watermark_embeddings"
            params = []
            
            if user_id:
                query += " WHERE user_id = ?"
                params.append(user_id)
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "id": row[0],
                    "user_id": row[1],
                    "session_id": row[2],
                    "content_hash": row[3],
                    "watermark_data": json.loads(row[4]),
                    "timestamp": row[5],
                    "metadata": json.loads(row[6]) if row[6] else {}
                }
                for row in results
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get watermark history: {e}")
            return []
    
    def verify_watermark(self, audio: np.ndarray, sample_rate: int,
                        expected_user_id: str) -> Dict[str, Any]:
        """Verify watermark for specific user"""
        detection_result = self.detect_watermark(audio, sample_rate)
        
        if not detection_result["detected"]:
            return {
                "verified": False,
                "confidence": 0.0,
                "message": "No watermark detected"
            }
        
        detected_user_id = detection_result["watermark_data"].get("user_id")
        confidence = detection_result["confidence"]
        
        if detected_user_id == expected_user_id:
            return {
                "verified": True,
                "confidence": confidence,
                "message": f"Watermark verified for user {expected_user_id}"
            }
        else:
            return {
                "verified": False,
                "confidence": confidence,
                "message": f"Watermark mismatch: expected {expected_user_id}, got {detected_user_id}"
            }







































