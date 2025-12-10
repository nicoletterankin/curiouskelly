"""
Quick Curriculum Audit
Analyzes Supabase content to identify gaps and priorities
"""

import json
import os
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()
load_dotenv("daily-lesson-marketing/.env")

try:
    from supabase import create_client, Client
except ImportError:
    print("❌ pip install supabase")
    exit(1)

# Supabase connection
url = os.environ.get("PUBLIC_SUPABASE_URL", "https://tvjalxxsyryjphkforjv.supabase.co")
key = os.environ.get("PUBLIC_SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR2amFseHhzeXJ5anBoa2Zvcmp2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NjM5MTksImV4cCI6MjA3OTEzOTkxOX0.VFrBs9sWkIgfFNpavQHxo0vSy6tkICpSbuj_TWvGHxI")

supabase: Client = create_client(url, key)

def main():
    print("")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║   📊 CURRICULUM AUDIT                                             ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print("")
    
    # 1. Core Lessons Summary
    print("📚 CORE LESSONS")
    print("-" * 50)
    lessons = supabase.table("core_lessons").select("*").execute()
    print(f"   Total lessons: {len(lessons.data)}")
    
    # Count by field completeness
    with_universal_truth = sum(1 for l in lessons.data if l.get("universal_truth"))
    with_marketing = sum(1 for l in lessons.data if l.get("marketing_headline"))
    with_objectives = sum(1 for l in lessons.data if l.get("learning_objectives"))
    
    print(f"   With universal_truth: {with_universal_truth} ({100*with_universal_truth/len(lessons.data):.0f}%)")
    print(f"   With marketing copy: {with_marketing} ({100*with_marketing/len(lessons.data):.0f}%)")
    print(f"   With learning objectives: {with_objectives} ({100*with_objectives/len(lessons.data):.0f}%)")
    
    # 2. Lesson Atoms Analysis
    print("")
    print("🧬 LESSON ATOMS")
    print("-" * 50)
    
    # Count atoms per lesson
    atoms = supabase.table("lesson_atoms").select("core_lesson_id, archetype, phase").execute()
    print(f"   Total atoms: {len(atoms.data)}")
    
    # Group by lesson
    atoms_per_lesson = defaultdict(list)
    for atom in atoms.data:
        atoms_per_lesson[atom["core_lesson_id"]].append(atom)
    
    print(f"   Lessons with atoms: {len(atoms_per_lesson)}")
    
    # Atoms count distribution
    atom_counts = [len(v) for v in atoms_per_lesson.values()]
    if atom_counts:
        print(f"   Min atoms per lesson: {min(atom_counts)}")
        print(f"   Max atoms per lesson: {max(atom_counts)}")
        print(f"   Avg atoms per lesson: {sum(atom_counts)/len(atom_counts):.1f}")
    
    # Archetype distribution
    archetypes = defaultdict(int)
    for atom in atoms.data:
        archetypes[atom.get("archetype", "Unknown")] += 1
    
    print("")
    print("   📊 Archetypes:")
    for arch, count in sorted(archetypes.items(), key=lambda x: -x[1])[:10]:
        print(f"      {arch}: {count}")
    
    # Phase distribution
    phases = defaultdict(int)
    for atom in atoms.data:
        phases[atom.get("phase", "Unknown")] += 1
    
    print("")
    print("   📊 Phases:")
    for phase, count in sorted(phases.items()):
        print(f"      {phase}: {count}")
    
    # 3. Lesson Shards Analysis
    print("")
    print("🎯 LESSON SHARDS (Age/Region Variants)")
    print("-" * 50)
    
    # Sample shards (don't load all 38K)
    shards_sample = supabase.table("lesson_shards").select("age, region, tone").limit(1000).execute()
    print(f"   Sample size: {len(shards_sample.data)}")
    
    # Age distribution
    ages = defaultdict(int)
    regions = defaultdict(int)
    tones = defaultdict(int)
    
    for shard in shards_sample.data:
        ages[shard.get("age", "?")] += 1
        regions[shard.get("region", "?")] += 1
        tones[shard.get("tone", "?")] += 1
    
    print("   📊 Ages (sample):")
    for age, count in sorted(ages.items()):
        print(f"      Age {age}: {count}")
    
    print("   📊 Regions:")
    for region, count in sorted(regions.items()):
        print(f"      {region}: {count}")
    
    print("   📊 Tones:")
    for tone, count in sorted(tones.items()):
        print(f"      {tone}: {count}")
    
    # 4. Gap Analysis
    print("")
    print("⚠️ GAP ANALYSIS")
    print("-" * 50)
    
    # Lessons without atoms
    lesson_ids = set(l["id"] for l in lessons.data)
    lessons_with_atoms = set(atoms_per_lesson.keys())
    lessons_without_atoms = lesson_ids - lessons_with_atoms
    
    print(f"   Lessons WITHOUT atoms: {len(lessons_without_atoms)}")
    if lessons_without_atoms:
        missing_days = []
        for l in lessons.data:
            if l["id"] in lessons_without_atoms:
                missing_days.append(l.get("day_number", "?"))
        print(f"   Days missing atoms: {sorted(missing_days)[:20]}{'...' if len(missing_days) > 20 else ''}")
    
    # Lessons with few atoms
    thin_lessons = [(lid, len(atoms)) for lid, atoms in atoms_per_lesson.items() if len(atoms) < 10]
    print(f"   Lessons with < 10 atoms: {len(thin_lessons)}")
    
    # 5. Quick Wins
    print("")
    print("🎯 QUICK WINS")
    print("-" * 50)
    print("   1. Fill lessons with 0 atoms (high priority)")
    print("   2. Add missing phases (Wisdom phase often missing)")
    print("   3. Ensure each lesson has 3+ archetypes")
    print("   4. Review thin lessons (< 10 atoms)")
    print("")
    
    # Summary
    print("=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    completeness = len(lessons_with_atoms) / len(lessons.data) * 100 if lessons.data else 0
    print(f"   Core Lessons:        {len(lessons.data)}")
    print(f"   Lesson Atoms:        {len(atoms.data)}")
    print(f"   Unique Archetypes:   {len(archetypes)}")
    print(f"   Coverage:            {completeness:.1f}%")
    print("=" * 60)
    print("")

if __name__ == "__main__":
    main()







