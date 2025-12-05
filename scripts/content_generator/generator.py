"""
Content Generator - Core Generation Logic
Uses OpenAI API to generate lesson content
"""

import json
import time
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

try:
    import openai
except ImportError:
    openai = None

try:
    from supabase import create_client, Client
except ImportError:
    print("❌ pip install supabase openai")
    
from .config import (
    OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY,
    ARCHETYPES, PHASES, AGE_BUCKETS, TONES, LANGUAGES,
    RATE_LIMIT_DELAY
)
from .prompts import (
    get_atom_generation_prompt,
    get_age_variant_prompt,
    get_translation_prompt
)


class ContentGenerator:
    """Generates lesson content using AI."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.stats = {
            "atoms_generated": 0,
            "shards_generated": 0,
            "translations_generated": 0,
            "errors": 0,
            "api_calls": 0,
        }
        
        # Initialize OpenAI
        if openai and OPENAI_API_KEY:
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        else:
            print("⚠️ OpenAI not configured. Set OPENAI_API_KEY environment variable.")
            self.client = None
        
        # Initialize Supabase
        if not dry_run:
            self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        else:
            self.supabase = None
    
    def _call_openai(self, prompt: str, max_retries: int = 3) -> Optional[dict]:
        """Call OpenAI API with retry logic."""
        if not self.client:
            return None
        
        for attempt in range(max_retries):
            try:
                self.stats["api_calls"] += 1
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",  # Fast and cost-effective
                    messages=[
                        {"role": "system", "content": "You are a JSON generator. Output ONLY valid JSON, no markdown, no explanation."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000,
                )
                
                content = response.choices[0].message.content.strip()
                
                # Clean up response - remove markdown code blocks if present
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()
                
                # Parse JSON
                return json.loads(content)
                
            except json.JSONDecodeError as e:
                print(f"   ⚠️ JSON parse error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
            except Exception as e:
                print(f"   ⚠️ API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        self.stats["errors"] += 1
        return None
    
    def generate_atom(self, lesson: dict, archetype: str, phase: str) -> Optional[dict]:
        """Generate a single lesson atom."""
        prompt = get_atom_generation_prompt(lesson, archetype, phase)
        content = self._call_openai(prompt)
        
        if content:
            atom = {
                "id": str(uuid.uuid4()),
                "core_lesson_id": lesson["id"],
                "archetype": archetype,
                "phase": phase,
                "content": content,
                "created_at": datetime.utcnow().isoformat(),
            }
            self.stats["atoms_generated"] += 1
            return atom
        return None
    
    def generate_shard(self, lesson: dict, age_bucket: dict, tone: str, region: str = "en") -> Optional[dict]:
        """Generate a single lesson shard (age variant)."""
        prompt = get_age_variant_prompt(lesson, age_bucket["range"], tone)
        content = self._call_openai(prompt)
        
        if content:
            shard = {
                "id": str(uuid.uuid4()),
                "core_lesson_id": lesson["id"],
                "age": age_bucket["age"],
                "region": region,
                "tone": tone,
                "birth_year": age_bucket["birth_year"],
                "script_content": content,
                "created_at": datetime.utcnow().isoformat(),
            }
            self.stats["shards_generated"] += 1
            return shard
        return None
    
    def translate_content(self, content: dict, target_lang: str) -> Optional[dict]:
        """Translate content to target language."""
        lang_name = LANGUAGES.get(target_lang, target_lang)
        prompt = get_translation_prompt(content, target_lang, lang_name)
        translated = self._call_openai(prompt)
        
        if translated:
            self.stats["translations_generated"] += 1
            return translated
        return None
    
    def upload_atoms(self, atoms: List[dict]) -> int:
        """Upload atoms to Supabase."""
        if self.dry_run or not atoms:
            return 0
        
        try:
            # Batch insert
            result = self.supabase.table("lesson_atoms").insert(atoms).execute()
            return len(result.data) if result.data else 0
        except Exception as e:
            print(f"   ❌ Upload error: {e}")
            return 0
    
    def upload_shards(self, shards: List[dict]) -> int:
        """Upload shards to Supabase."""
        if self.dry_run or not shards:
            return 0
        
        try:
            result = self.supabase.table("lesson_shards").insert(shards).execute()
            return len(result.data) if result.data else 0
        except Exception as e:
            print(f"   ❌ Upload error: {e}")
            return 0
    
    def generate_all_atoms_for_lesson(self, lesson: dict, archetypes: List[str] = None, phases: List[str] = None) -> List[dict]:
        """Generate all atoms for a single lesson."""
        archetypes = archetypes or ARCHETYPES
        phases = phases or PHASES
        
        atoms = []
        total = len(archetypes) * len(phases)
        
        for i, archetype in enumerate(archetypes):
            for j, phase in enumerate(phases):
                current = i * len(phases) + j + 1
                print(f"      [{current}/{total}] {archetype} - {phase}")
                
                atom = self.generate_atom(lesson, archetype, phase)
                if atom:
                    atoms.append(atom)
                
                time.sleep(RATE_LIMIT_DELAY)
        
        return atoms
    
    def generate_all_shards_for_lesson(self, lesson: dict, age_buckets: List[dict] = None, tones: List[str] = None) -> List[dict]:
        """Generate all age/tone variants for a single lesson."""
        age_buckets = age_buckets or AGE_BUCKETS
        tones = tones or TONES
        
        shards = []
        total = len(age_buckets) * len(tones)
        
        for i, age_bucket in enumerate(age_buckets):
            for j, tone in enumerate(tones):
                current = i * len(tones) + j + 1
                print(f"      [{current}/{total}] Age {age_bucket['age']} - {tone}")
                
                shard = self.generate_shard(lesson, age_bucket, tone)
                if shard:
                    shards.append(shard)
                
                time.sleep(RATE_LIMIT_DELAY)
        
        return shards
    
    def generate_translations_for_shards(self, shards: List[dict], target_langs: List[str] = None) -> List[dict]:
        """Generate translations for existing shards."""
        target_langs = target_langs or ["es", "fr"]
        
        translated_shards = []
        total = len(shards) * len(target_langs)
        
        for i, shard in enumerate(shards):
            for j, lang in enumerate(target_langs):
                current = i * len(target_langs) + j + 1
                print(f"      [{current}/{total}] Translating to {lang}")
                
                translated_content = self.translate_content(shard["script_content"], lang)
                if translated_content:
                    new_shard = shard.copy()
                    new_shard["id"] = str(uuid.uuid4())
                    new_shard["region"] = lang
                    new_shard["script_content"] = translated_content
                    translated_shards.append(new_shard)
                
                time.sleep(RATE_LIMIT_DELAY)
        
        return translated_shards
    
    def print_stats(self):
        """Print generation statistics."""
        print("\n" + "=" * 50)
        print("📊 GENERATION STATS")
        print("=" * 50)
        print(f"   Atoms generated:        {self.stats['atoms_generated']}")
        print(f"   Shards generated:       {self.stats['shards_generated']}")
        print(f"   Translations generated: {self.stats['translations_generated']}")
        print(f"   API calls made:         {self.stats['api_calls']}")
        print(f"   Errors:                 {self.stats['errors']}")
        print("=" * 50)





