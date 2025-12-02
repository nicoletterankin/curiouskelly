"""
Content Generator Configuration
"""

import os
from pathlib import Path

# Find project root and load .env manually
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

def load_env_file(env_path: Path) -> dict:
    """Load environment variables from file."""
    env_vars = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
                    os.environ[key.strip()] = value.strip()
    return env_vars

# Load environment variables
_env = load_env_file(ENV_FILE)
_env.update(load_env_file(PROJECT_ROOT / "daily-lesson-marketing" / ".env"))

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or _env.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Supabase
SUPABASE_URL = os.environ.get("PUBLIC_SUPABASE_URL", "https://tvjalxxsyryjphkforjv.supabase.co")
SUPABASE_KEY = os.environ.get("PUBLIC_SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", SUPABASE_KEY)

# Archetypes - personality profiles for content adaptation
ARCHETYPES = [
    "The Scientist",      # Analytical, evidence-based, methodical
    "The Explorer",       # Curious, adventurous, discovery-focused
    "The Storyteller",    # Narrative, emotional, relatable
    "The Survivor",       # Practical, resilient, real-world applications
    "The MacGyver",       # Creative problem-solver, hands-on, innovative
    "The Empath",         # Emotionally intelligent, relational, compassionate
    "The Rebel",          # Questioning, challenging assumptions, unconventional
    "The Architect",      # Systems thinker, structured, big-picture
    "The Mystic",         # Philosophical, meaning-seeking, contemplative
    "The Diplomat",       # Balanced, multiple perspectives, bridge-builder
]

# Phases - lesson structure
PHASES = [
    "Hook",      # Opening hook to capture attention
    "Fact1",     # First fascinating fact
    "Fact2",     # Second fact, building depth
    "Fact3",     # Third fact, surprising or delightful
    "Wisdom",    # Closing wisdom/reflection
]

# Age buckets for variants
AGE_BUCKETS = [
    {"age": 4, "label": "early_childhood", "range": "2-5", "birth_year": 2021},
    {"age": 9, "label": "elementary", "range": "6-12", "birth_year": 2016},
    {"age": 15, "label": "teen", "range": "13-17", "birth_year": 2010},
    {"age": 26, "label": "young_adult", "range": "18-35", "birth_year": 1999},
    {"age": 45, "label": "midlife", "range": "36-60", "birth_year": 1980},
    {"age": 72, "label": "wisdom_years", "range": "61-102", "birth_year": 1953},
]

# Tones for shards
TONES = ["curious", "playful", "serious", "warm", "inspiring"]

# Languages
LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
}

# Rate limiting
RATE_LIMIT_DELAY = 0.5  # seconds between API calls
BATCH_SIZE = 10  # lessons per batch

