#!/usr/bin/env python3
"""
Security Hardening Script for TTS System
Implements runtime hardening, input validation, and security monitoring
"""

import os
import sys
import json
import hashlib
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
import re
import numpy as np
import librosa
from dataclasses import dataclass
import resource
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    max_text_length: int = 1000
    max_audio_duration: int = 300  # 5 minutes
    max_memory_usage: int = 1024  # MB
    max_cpu_usage: float = 80.0  # Percentage
    allowed_languages: List[str] = None
    prohibited_patterns: List[str] = None
    watermark_strength: float = 0.001
    log_all_requests: bool = True
    
    def __post_init__(self):
        if self.allowed_languages is None:
            self.allowed_languages = ["en"]
        if self.prohibited_patterns is None:
            self.prohibited_patterns = [
                r"\b(hate|violence|discrimination)\b",
                r"\b(illegal|harmful|offensive)\b",
                r"\b(fake|false|misinformation)\b"
            ]

class InputValidator:
    """Validates and sanitizes input text"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.InputValidator")
    
    def validate_text(self, text: str) -> Dict[str, Any]:
        """Validate input text for security and policy compliance"""
        violations = []
        
        # Length validation
        if len(text) > self.config.max_text_length:
            violations.append({
                "type": "length_violation",
                "message": f"Text length {len(text)} exceeds maximum {self.config.max_text_length}",
                "severity": "error"
            })
        
        # Character validation
        if not self.is_safe_text(text):
            violations.append({
                "type": "character_violation",
                "message": "Text contains potentially dangerous characters",
                "severity": "error"
            })
        
        # Content policy validation
        policy_violations = self.check_content_policy(text)
        violations.extend(policy_violations)
        
        # Language validation
        if not self.is_allowed_language(text):
            violations.append({
                "type": "language_violation",
                "message": "Text language not allowed",
                "severity": "error"
            })
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "metadata": {
                "length": len(text),
                "hash": hashlib.sha256(text.encode()).hexdigest(),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def is_safe_text(self, text: str) -> bool:
        """Check if text contains only safe characters"""
        # Allow alphanumeric, spaces, and common punctuation
        safe_pattern = r'^[a-zA-Z0-9\s.,!?;:\'"()-]+$'
        return bool(re.match(safe_pattern, text))
    
    def check_content_policy(self, text: str) -> List[Dict[str, Any]]:
        """Check text against content policy"""
        violations = []
        
        for pattern in self.config.prohibited_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append({
                    "type": "content_policy_violation",
                    "message": f"Text matches prohibited pattern: {pattern}",
                    "severity": "error"
                })
        
        return violations
    
    def is_allowed_language(self, text: str) -> bool:
        """Simple language detection (replace with proper library)"""
        # This is a placeholder - in production, use langdetect or similar
        return True  # Assume English for now
    
    def sanitize_text(self, text: str) -> str:
        """Sanitize text by removing dangerous characters"""
        # Remove potentially dangerous characters
        dangerous_chars = ["'", '"', ";", "\\", "/", "<", ">", "&"]
        sanitized = text
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "")
        
        return sanitized.strip()

class ResourceMonitor:
    """Monitors system resources to prevent abuse"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ResourceMonitor")
        self.set_memory_limits()
    
    def set_memory_limits(self):
        """Set memory limits for the process"""
        try:
            # Set memory limit to 1GB
            resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, -1))
            # Set data segment limit to 512MB
            resource.setrlimit(resource.RLIMIT_DATA, (512 * 1024 * 1024, -1))
            self.logger.info("Memory limits set successfully")
        except Exception as e:
            self.logger.error(f"Failed to set memory limits: {e}")
    
    def check_resources(self) -> Dict[str, Any]:
        """Check current resource usage"""
        process = psutil.Process()
        
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        
        cpu_percent = process.cpu_percent()
        
        # Check if limits are exceeded
        memory_exceeded = memory_usage_mb > self.config.max_memory_usage
        cpu_exceeded = cpu_percent > self.config.max_cpu_usage
        
        if memory_exceeded or cpu_exceeded:
            self.logger.warning(f"Resource limits exceeded: Memory={memory_usage_mb:.1f}MB, CPU={cpu_percent:.1f}%")
        
        return {
            "memory_usage_mb": memory_usage_mb,
            "cpu_percent": cpu_percent,
            "memory_exceeded": memory_exceeded,
            "cpu_exceeded": cpu_exceeded,
            "timestamp": datetime.now().isoformat()
        }

class AudioWatermarker:
    """Embeds and detects watermarks in audio"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.AudioWatermarker")
    
    def embed_watermark(self, audio: np.ndarray, user_id: str, metadata: Dict[str, Any]) -> np.ndarray:
        """Embed watermark in audio"""
        try:
            # Generate watermark signal
            watermark_signal = self.generate_watermark_signal(
                len(audio), user_id, metadata
            )
            
            # Embed watermark
            watermarked_audio = audio + (watermark_signal * self.config.watermark_strength)
            
            # Ensure no clipping
            watermarked_audio = np.clip(watermarked_audio, -1.0, 1.0)
            
            self.logger.info(f"Watermark embedded for user {user_id}")
            return watermarked_audio
            
        except Exception as e:
            self.logger.error(f"Failed to embed watermark: {e}")
            return audio
    
    def generate_watermark_signal(self, length: int, user_id: str, metadata: Dict[str, Any]) -> np.ndarray:
        """Generate watermark signal based on user ID and metadata"""
        # Create unique seed from user ID and metadata
        seed_data = f"{user_id}:{metadata.get('timestamp', '0')}:{metadata.get('session_id', '0')}"
        seed = int(hashlib.sha256(seed_data.encode()).hexdigest()[:8], 16)
        
        # Generate pseudo-random signal
        np.random.seed(seed)
        watermark = np.random.normal(0, 1, length)
        
        # Apply frequency shaping to make it less audible
        # Emphasize frequencies above 8kHz (less audible to humans)
        freqs = np.fft.fftfreq(length)
        freq_mask = np.abs(freqs) > 0.1  # Above 8kHz for 44.1kHz sample rate
        watermark_fft = np.fft.fft(watermark)
        watermark_fft *= freq_mask.astype(float)
        watermark = np.real(np.fft.ifft(watermark_fft))
        
        return watermark
    
    def extract_watermark(self, audio: np.ndarray, user_id: str) -> Dict[str, Any]:
        """Extract watermark information from audio"""
        try:
            # Generate expected watermark
            expected_watermark = self.generate_watermark_signal(
                len(audio), user_id, {"timestamp": "0", "session_id": "0"}
            )
            
            # Calculate correlation
            correlation = np.corrcoef(audio, expected_watermark)[0, 1]
            
            return {
                "detected": correlation > 0.1,  # Threshold for detection
                "confidence": correlation,
                "user_id": user_id if correlation > 0.1 else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract watermark: {e}")
            return {"detected": False, "confidence": 0.0, "user_id": None}

class SecurityLogger:
    """Logs security events and tracks usage"""
    
    def __init__(self, db_path: str = "security_log.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.SecurityLogger")
        self.init_database()
    
    def init_database(self):
        """Initialize security logging database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    ip_address TEXT,
                    timestamp DATETIME NOT NULL,
                    details TEXT,
                    severity TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS content_generation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    text_input TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    policy_compliance BOOLEAN NOT NULL,
                    watermark_embedded BOOLEAN NOT NULL,
                    metadata TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            self.logger.info("Security database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize security database: {e}")
    
    def log_security_event(self, event_type: str, user_id: str = None, 
                          ip_address: str = None, details: str = None, 
                          severity: str = "info"):
        """Log a security event"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO security_events 
                (event_type, user_id, ip_address, timestamp, details, severity)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event_type, user_id, ip_address, 
                datetime.now(), details, severity
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Security event logged: {event_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to log security event: {e}")
    
    def log_content_generation(self, user_id: str, text_input: str, 
                              policy_compliance: bool, watermark_embedded: bool,
                              metadata: Dict[str, Any] = None):
        """Log content generation event"""
        try:
            content_hash = hashlib.sha256(text_input.encode()).hexdigest()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO content_generation 
                (user_id, content_hash, text_input, timestamp, 
                 policy_compliance, watermark_embedded, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, content_hash, text_input, datetime.now(),
                policy_compliance, watermark_embedded, 
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Content generation logged for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to log content generation: {e}")

class SecureTTSProcessor:
    """Main secure TTS processor with all security features"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.logger = logging.getLogger(f"{__name__}.SecureTTSProcessor")
        
        # Initialize security components
        self.input_validator = InputValidator(self.config)
        self.resource_monitor = ResourceMonitor(self.config)
        self.watermarker = AudioWatermarker(self.config)
        self.security_logger = SecurityLogger()
        
        self.logger.info("Secure TTS Processor initialized")
    
    def process_text(self, text: str, user_id: str, ip_address: str = None) -> Dict[str, Any]:
        """Process text with full security validation"""
        try:
            # 1. Check system resources
            resource_status = self.resource_monitor.check_resources()
            if resource_status["memory_exceeded"] or resource_status["cpu_exceeded"]:
                self.security_logger.log_security_event(
                    "resource_limit_exceeded", user_id, ip_address,
                    f"Memory: {resource_status['memory_usage_mb']:.1f}MB, "
                    f"CPU: {resource_status['cpu_percent']:.1f}%",
                    "warning"
                )
                raise RuntimeError("System resource limits exceeded")
            
            # 2. Validate input
            validation_result = self.input_validator.validate_text(text)
            if not validation_result["valid"]:
                self.security_logger.log_security_event(
                    "input_validation_failed", user_id, ip_address,
                    f"Violations: {validation_result['violations']}",
                    "error"
                )
                raise ValueError("Input validation failed")
            
            # 3. Sanitize text
            sanitized_text = self.input_validator.sanitize_text(text)
            
            # 4. Log the request
            self.security_logger.log_security_event(
                "text_processing_request", user_id, ip_address,
                f"Text length: {len(sanitized_text)}", "info"
            )
            
            # 5. Generate audio (placeholder - replace with actual TTS)
            audio = self.generate_audio_placeholder(sanitized_text)
            
            # 6. Embed watermark
            metadata = {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "session_id": hashlib.md5(f"{user_id}:{datetime.now()}".encode()).hexdigest()[:8]
            }
            
            watermarked_audio = self.watermarker.embed_watermark(audio, user_id, metadata)
            
            # 7. Log content generation
            self.security_logger.log_content_generation(
                user_id, sanitized_text, True, True, metadata
            )
            
            return {
                "success": True,
                "audio": watermarked_audio,
                "metadata": metadata,
                "security_log": "Content generated and watermarked successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            self.security_logger.log_security_event(
                "processing_failed", user_id, ip_address, str(e), "error"
            )
            return {
                "success": False,
                "error": str(e),
                "security_log": "Processing failed due to security or system error"
            }
    
    def generate_audio_placeholder(self, text: str) -> np.ndarray:
        """Generate placeholder audio (replace with actual TTS)"""
        # This is a placeholder - replace with your actual TTS system
        duration = min(len(text) * 0.1, 10.0)  # 0.1 seconds per character, max 10 seconds
        sample_rate = 22050
        samples = int(duration * sample_rate)
        
        # Generate a simple tone (replace with actual TTS)
        frequency = 440 + (hash(text) % 200)  # Vary frequency based on text
        t = np.linspace(0, duration, samples)
        audio = 0.1 * np.sin(2 * np.pi * frequency * t)
        
        return audio

def main():
    """Main function for testing security hardening"""
    print("🔒 TTS Security Hardening Test")
    print("=" * 50)
    
    # Initialize secure processor
    config = SecurityConfig()
    processor = SecureTTSProcessor(config)
    
    # Test cases
    test_cases = [
        ("Hello world", "user1", "192.168.1.1"),
        ("This is a normal sentence", "user2", "192.168.1.2"),
        ("A" * 2000, "user3", "192.168.1.3"),  # Too long
        ("'; DROP TABLE users; --", "user4", "192.168.1.4"),  # SQL injection
        ("Generate hate speech", "user5", "192.168.1.5"),  # Prohibited content
    ]
    
    for text, user_id, ip in test_cases:
        print(f"\n🧪 Testing: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"   User: {user_id}, IP: {ip}")
        
        result = processor.process_text(text, user_id, ip)
        
        if result["success"]:
            print(f"   ✅ Success: {result['security_log']}")
        else:
            print(f"   ❌ Failed: {result['error']}")
    
    print(f"\n📊 Security logs saved to: security.log")
    print(f"📊 Security database: security_log.db")
    print(f"\n🔒 Security hardening test complete!")

if __name__ == "__main__":
    main()







































