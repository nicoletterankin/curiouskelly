"""
Content Policy Engine for TTS System
Implements content filtering, policy enforcement, and compliance monitoring
"""

import re
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import sqlite3
from enum import Enum

logger = logging.getLogger(__name__)

class PolicySeverity(Enum):
    """Policy violation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ContentCategory(Enum):
    """Content categories for policy enforcement"""
    SAFE = "safe"
    QUESTIONABLE = "questionable"
    PROHIBITED = "prohibited"
    ILLEGAL = "illegal"

@dataclass
class PolicyRule:
    """Individual policy rule definition"""
    name: str
    pattern: str
    severity: PolicySeverity
    category: ContentCategory
    description: str
    enabled: bool = True
    case_sensitive: bool = False

@dataclass
class ContentPolicy:
    """Main content policy configuration"""
    max_length: int = 1000
    min_length: int = 1
    allowed_languages: List[str] = field(default_factory=lambda: ["en"])
    prohibited_languages: List[str] = field(default_factory=list)
    age_rating: str = "G"  # G, PG, PG-13, R, NC-17
    commercial_use: bool = False
    rules: List[PolicyRule] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.rules:
            self.rules = self._create_default_rules()
    
    def _create_default_rules(self) -> List[PolicyRule]:
        """Create default policy rules"""
        return [
            PolicyRule(
                name="hate_speech",
                pattern=r"\b(hate|racist|discriminat|bigot|prejudice)\b",
                severity=PolicySeverity.ERROR,
                category=ContentCategory.PROHIBITED,
                description="Hate speech and discrimination"
            ),
            PolicyRule(
                name="violence",
                pattern=r"\b(kill|murder|violence|weapon|attack|harm)\b",
                severity=PolicySeverity.WARNING,
                category=ContentCategory.QUESTIONABLE,
                description="Violent content"
            ),
            PolicyRule(
                name="adult_content",
                pattern=r"\b(sex|porn|adult|explicit|nude)\b",
                severity=PolicySeverity.ERROR,
                category=ContentCategory.PROHIBITED,
                description="Adult content"
            ),
            PolicyRule(
                name="illegal_activities",
                pattern=r"\b(drug|illegal|crime|fraud|theft)\b",
                severity=PolicySeverity.ERROR,
                category=ContentCategory.ILLEGAL,
                description="Illegal activities"
            ),
            PolicyRule(
                name="misinformation",
                pattern=r"\b(fake|false|conspiracy|hoax|lie)\b",
                severity=PolicySeverity.WARNING,
                category=ContentCategory.QUESTIONABLE,
                description="Potential misinformation"
            ),
            PolicyRule(
                name="personal_info",
                pattern=r"\b(ssn|social security|credit card|password|pin)\b",
                severity=PolicySeverity.ERROR,
                category=ContentCategory.PROHIBITED,
                description="Personal information"
            ),
            PolicyRule(
                name="spam_patterns",
                pattern=r"\b(buy now|click here|free money|guaranteed)\b",
                severity=PolicySeverity.WARNING,
                category=ContentCategory.QUESTIONABLE,
                description="Spam-like content"
            )
        ]

class ContentPolicyEngine:
    """Main content policy enforcement engine"""
    
    def __init__(self, policy: ContentPolicy = None, db_path: str = "policy_log.db"):
        self.policy = policy or ContentPolicy()
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.ContentPolicyEngine")
        self._init_database()
    
    def _init_database(self):
        """Initialize policy logging database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS policy_violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    content_hash TEXT,
                    rule_name TEXT,
                    severity TEXT,
                    category TEXT,
                    timestamp DATETIME,
                    content_preview TEXT,
                    metadata TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS content_approvals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    content_hash TEXT,
                    approved BOOLEAN,
                    reviewer_id TEXT,
                    timestamp DATETIME,
                    notes TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            self.logger.info("Policy database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize policy database: {e}")
    
    def validate_content(self, text: str, user_id: str = None, 
                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate content against policy rules"""
        
        # Basic validation
        basic_validation = self._validate_basic_requirements(text)
        if not basic_validation["valid"]:
            return basic_validation
        
        # Rule-based validation
        rule_violations = self._check_policy_rules(text)
        
        # Language validation
        language_validation = self._validate_language(text)
        
        # Combine results
        all_violations = (
            basic_validation.get("violations", []) +
            rule_violations +
            language_validation.get("violations", [])
        )
        
        # Determine overall result
        is_valid = len(all_violations) == 0
        severity = self._get_highest_severity(all_violations)
        
        # Log violations
        if all_violations:
            self._log_violations(user_id, text, all_violations, context)
        
        return {
            "valid": is_valid,
            "violations": all_violations,
            "severity": severity,
            "metadata": {
                "length": len(text),
                "hash": hashlib.sha256(text.encode()).hexdigest(),
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id
            }
        }
    
    def _validate_basic_requirements(self, text: str) -> Dict[str, Any]:
        """Validate basic content requirements"""
        violations = []
        
        # Length validation
        if len(text) < self.policy.min_length:
            violations.append({
                "type": "length_violation",
                "message": f"Text too short (minimum {self.policy.min_length} characters)",
                "severity": PolicySeverity.ERROR.value,
                "rule": "basic_requirements"
            })
        
        if len(text) > self.policy.max_length:
            violations.append({
                "type": "length_violation",
                "message": f"Text too long (maximum {self.policy.max_length} characters)",
                "severity": PolicySeverity.ERROR.value,
                "rule": "basic_requirements"
            })
        
        # Character validation
        if not self._is_safe_text(text):
            violations.append({
                "type": "character_violation",
                "message": "Text contains potentially dangerous characters",
                "severity": PolicySeverity.ERROR.value,
                "rule": "basic_requirements"
            })
        
        return {
            "valid": len(violations) == 0,
            "violations": violations
        }
    
    def _check_policy_rules(self, text: str) -> List[Dict[str, Any]]:
        """Check content against policy rules"""
        violations = []
        
        for rule in self.policy.rules:
            if not rule.enabled:
                continue
            
            # Check if pattern matches
            flags = 0 if rule.case_sensitive else re.IGNORECASE
            if re.search(rule.pattern, text, flags):
                violations.append({
                    "type": "policy_violation",
                    "message": f"Content violates rule: {rule.name}",
                    "severity": rule.severity.value,
                    "category": rule.category.value,
                    "rule": rule.name,
                    "description": rule.description,
                    "pattern": rule.pattern
                })
        
        return violations
    
    def _validate_language(self, text: str) -> Dict[str, Any]:
        """Validate language requirements"""
        violations = []
        
        # Simple language detection (replace with proper library)
        detected_language = self._detect_language(text)
        
        if detected_language in self.policy.prohibited_languages:
            violations.append({
                "type": "language_violation",
                "message": f"Language '{detected_language}' is prohibited",
                "severity": PolicySeverity.ERROR.value,
                "rule": "language_policy"
            })
        
        if detected_language not in self.policy.allowed_languages:
            violations.append({
                "type": "language_violation",
                "message": f"Language '{detected_language}' not in allowed languages",
                "severity": PolicySeverity.WARNING.value,
                "rule": "language_policy"
            })
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "detected_language": detected_language
        }
    
    def _is_safe_text(self, text: str) -> bool:
        """Check if text contains only safe characters"""
        # Allow alphanumeric, spaces, and common punctuation
        safe_pattern = r'^[a-zA-Z0-9\s.,!?;:\'"()-]+$'
        return bool(re.match(safe_pattern, text))
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection (replace with proper library)"""
        # This is a placeholder - in production, use langdetect or similar
        # For now, assume English
        return "en"
    
    def _get_highest_severity(self, violations: List[Dict[str, Any]]) -> str:
        """Get the highest severity level from violations"""
        severity_order = {
            PolicySeverity.INFO.value: 0,
            PolicySeverity.WARNING.value: 1,
            PolicySeverity.ERROR.value: 2,
            PolicySeverity.CRITICAL.value: 3
        }
        
        if not violations:
            return PolicySeverity.INFO.value
        
        highest_severity = max(
            violations,
            key=lambda v: severity_order.get(v.get("severity", "info"), 0)
        )
        
        return highest_severity.get("severity", PolicySeverity.INFO.value)
    
    def _log_violations(self, user_id: str, text: str, violations: List[Dict[str, Any]], 
                       context: Dict[str, Any] = None):
        """Log policy violations to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            
            for violation in violations:
                cursor.execute("""
                    INSERT INTO policy_violations 
                    (user_id, content_hash, rule_name, severity, category, 
                     timestamp, content_preview, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    content_hash,
                    violation.get("rule", "unknown"),
                    violation.get("severity", "info"),
                    violation.get("category", "unknown"),
                    datetime.now(),
                    text[:100],  # First 100 characters
                    json.dumps(context) if context else None
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Logged {len(violations)} policy violations for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to log policy violations: {e}")
    
    def get_violation_history(self, user_id: str = None, 
                            severity: str = None) -> List[Dict[str, Any]]:
        """Get violation history from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM policy_violations WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "id": row[0],
                    "user_id": row[1],
                    "content_hash": row[2],
                    "rule_name": row[3],
                    "severity": row[4],
                    "category": row[5],
                    "timestamp": row[6],
                    "content_preview": row[7],
                    "metadata": json.loads(row[8]) if row[8] else {}
                }
                for row in results
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get violation history: {e}")
            return []
    
    def approve_content(self, content_hash: str, user_id: str, 
                       reviewer_id: str, notes: str = None) -> bool:
        """Approve content for generation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO content_approvals 
                (user_id, content_hash, approved, reviewer_id, timestamp, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id, content_hash, True, reviewer_id, 
                datetime.now(), notes
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Content {content_hash} approved by {reviewer_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to approve content: {e}")
            return False
    
    def is_content_approved(self, content_hash: str) -> bool:
        """Check if content is approved"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT approved FROM content_approvals 
                WHERE content_hash = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (content_hash,))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else False
            
        except Exception as e:
            self.logger.error(f"Failed to check content approval: {e}")
            return False







































