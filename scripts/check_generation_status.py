#!/usr/bin/env python3
"""
Check Generation Status
Shows current content counts in Supabase
"""

import os
from pathlib import Path

# Load env
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    with open(ENV_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

from supabase import create_client

SUPABASE_URL = os.environ.get("PUBLIC_SUPABASE_URL", "https://tvjalxxsyryjphkforjv.supabase.co")
SUPABASE_KEY = os.environ.get("PUBLIC_SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def main():
    print("")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║   📊 CONTENT GENERATION STATUS                                    ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print("")
    
    # Core lessons
    lessons = supabase.table("core_lessons").select("id", count="exact").execute()
    print(f"📚 Core Lessons: {lessons.count}")
    
    # Atoms by archetype
    print("\n🧬 Lesson Atoms:")
    atoms = supabase.table("lesson_atoms").select("archetype", count="exact").execute()
    print(f"   Total: {atoms.count}")
    
    # Get archetype breakdown (sample)
    archetypes = supabase.table("lesson_atoms")\
        .select("archetype")\
        .limit(10000)\
        .execute()
    
    if archetypes.data:
        from collections import Counter
        arch_counts = Counter(a["archetype"] for a in archetypes.data)
        for arch, count in sorted(arch_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"      {arch}: {count}")
        if len(arch_counts) > 5:
            print(f"      ... and {len(arch_counts) - 5} more archetypes")
    
    # Unique lessons with atoms
    lessons_with_atoms = supabase.table("lesson_atoms")\
        .select("core_lesson_id")\
        .execute()
    unique_lessons = len(set(a["core_lesson_id"] for a in (lessons_with_atoms.data or [])))
    print(f"   Lessons covered: {unique_lessons}/365 ({100*unique_lessons/365:.1f}%)")
    
    # Shards
    print("\n👤 Lesson Shards:")
    shards = supabase.table("lesson_shards").select("region", count="exact").execute()
    print(f"   Total: {shards.count}")
    
    # Region breakdown
    regions = supabase.table("lesson_shards")\
        .select("region")\
        .limit(10000)\
        .execute()
    
    if regions.data:
        from collections import Counter
        region_counts = Counter(r["region"] for r in regions.data)
        for region, count in sorted(region_counts.items()):
            flag = {"en": "🇺🇸", "es": "🇪🇸", "fr": "🇫🇷"}.get(region, "🌐")
            print(f"      {flag} {region}: {count}")
    
    # Progress bars
    print("\n📈 Progress:")
    
    atoms_target = 365 * 15  # 15 atoms per lesson (essential)
    atoms_current = atoms.count or 0
    atoms_pct = min(100, 100 * atoms_current / atoms_target)
    atoms_bar = "█" * int(atoms_pct / 5) + "░" * (20 - int(atoms_pct / 5))
    print(f"   Atoms:  [{atoms_bar}] {atoms_pct:.1f}% ({atoms_current}/{atoms_target})")
    
    shards_target = 365 * 6  # 6 shards per lesson (essential)
    shards_current = shards.count or 0
    shards_pct = min(100, 100 * shards_current / shards_target)
    shards_bar = "█" * int(shards_pct / 5) + "░" * (20 - int(shards_pct / 5))
    print(f"   Shards: [{shards_bar}] {shards_pct:.1f}% ({shards_current}/{shards_target})")
    
    # Translations progress
    if regions.data:
        en_count = region_counts.get("en", 0)
        es_count = region_counts.get("es", 0)
        fr_count = region_counts.get("fr", 0)
        trans_target = en_count * 2  # ES + FR
        trans_current = es_count + fr_count
        trans_pct = min(100, 100 * trans_current / trans_target) if trans_target > 0 else 0
        trans_bar = "█" * int(trans_pct / 5) + "░" * (20 - int(trans_pct / 5))
        print(f"   Trans:  [{trans_bar}] {trans_pct:.1f}% ({trans_current}/{trans_target})")
    
    print("")


if __name__ == "__main__":
    main()




