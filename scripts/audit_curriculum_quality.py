"""
Audit Curriculum Quality - Find mismatches between titles and learning objectives

This script identifies potential data quality issues in the Learn track curriculum.
"""

import json
from pathlib import Path
from collections import defaultdict

def load_calendar():
    """Load the 365-day calendar."""
    calendar_path = Path(__file__).parent.parent / "lessons" / "365_day_calendar.json"
    with open(calendar_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_keywords(text):
    """Extract key topic words from text."""
    # Common words to ignore
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why',
        'how', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
        'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 's', 't', 'just', 'don', 'now', 'our', 'us', 'we',
        'you', 'your', 'they', 'them', 'their', 'it', 'its', 'he', 'she',
        'him', 'her', 'his', 'hers', 'my', 'me', 'i', 'about', 'up', 'down',
        'out', 'off', 'over', 'every', 'any', 'both', 'each', 'while',
        'learn', 'learning', 'understand', 'explore', 'discover', 'know',
        'life', 'world', 'people', 'things', 'way', 'ways', 'makes', 'make',
        'helps', 'help', 'allows', 'allow', 'creates', 'create', 'provides'
    }
    
    words = text.lower().replace('-', ' ').replace(':', ' ').replace('?', ' ')
    words = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in words)
    words = set(words.split()) - stopwords
    return words

def check_topic_alignment(title, objective):
    """Check if title and objective seem to be about the same topic."""
    title_words = extract_keywords(title)
    obj_words = extract_keywords(objective)
    
    # Check for overlap
    common = title_words & obj_words
    
    # If there's meaningful overlap, they're probably aligned
    if len(common) >= 1:
        return True, common
    
    # Check for related concept pairs
    related_pairs = [
        ({'sun', 'solar'}, {'sun', 'solar', 'light', 'energy', 'star'}),
        ({'moon', 'lunar'}, {'moon', 'lunar', 'tides', 'phases'}),
        ({'water', 'ocean', 'river', 'lake'}, {'water', 'ocean', 'river', 'lake', 'liquid', 'h2o'}),
        ({'forest', 'tree', 'trees'}, {'forest', 'tree', 'trees', 'wood', 'woods', 'oxygen'}),
        ({'desert'}, {'desert', 'sand', 'dry', 'arid', 'cactus'}),
        ({'coral', 'reef'}, {'coral', 'reef', 'ocean', 'marine', 'fish'}),
        ({'cave', 'caves'}, {'cave', 'caves', 'underground', 'dark'}),
        ({'insect', 'insects', 'bug', 'bugs'}, {'insect', 'insects', 'bug', 'bugs', 'butterfly', 'ant'}),
        ({'bird', 'birds'}, {'bird', 'birds', 'feather', 'feathers', 'fly', 'wing'}),
        ({'fish', 'fishes'}, {'fish', 'fishes', 'gills', 'swim', 'ocean', 'sea'}),
    ]
    
    for title_set, obj_set in related_pairs:
        if title_words & title_set and obj_words & obj_set:
            return True, title_words & title_set
    
    return False, set()

def audit_curriculum():
    """Run the full audit."""
    print("")
    print("=" * 70)
    print("   CURRICULUM DATA QUALITY AUDIT")
    print("=" * 70)
    print("")
    
    data = load_calendar()
    lessons = data.get("lessons", [])
    
    print(f"Total lessons: {len(lessons)}")
    print("")
    
    mismatches = []
    duplicates = defaultdict(list)
    missing_fields = []
    
    for lesson in lessons:
        day = lesson.get("day", 0)
        title = lesson.get("title", "")
        objective = lesson.get("learning_objective", "")
        
        # Check for missing fields
        if not title:
            missing_fields.append((day, "title"))
        if not objective:
            missing_fields.append((day, "learning_objective"))
        
        # Track duplicates
        duplicates[title.lower()].append(day)
        
        # Check alignment
        if title and objective:
            aligned, common = check_topic_alignment(title, objective)
            if not aligned:
                mismatches.append({
                    "day": day,
                    "title": title,
                    "objective": objective[:80] + "..." if len(objective) > 80 else objective
                })
    
    # Report mismatches
    print("-" * 70)
    print("POTENTIAL TITLE/OBJECTIVE MISMATCHES")
    print("-" * 70)
    
    if mismatches:
        print(f"Found {len(mismatches)} potential mismatches:\n")
        for m in mismatches:
            print(f"Day {m['day']:3d}: {m['title']}")
            print(f"        Objective: {m['objective']}")
            print("")
    else:
        print("No obvious mismatches found!")
    
    # Report duplicates
    print("-" * 70)
    print("DUPLICATE TITLES")
    print("-" * 70)
    
    dup_count = 0
    for title, days in duplicates.items():
        if len(days) > 1:
            dup_count += 1
            print(f"'{title}' appears on days: {days}")
    
    if dup_count == 0:
        print("No duplicate titles found!")
    else:
        print(f"\nTotal: {dup_count} duplicate titles")
    
    # Report missing
    print("")
    print("-" * 70)
    print("MISSING FIELDS")
    print("-" * 70)
    
    if missing_fields:
        for day, field in missing_fields:
            print(f"Day {day}: missing {field}")
    else:
        print("All fields present!")
    
    # Summary
    print("")
    print("=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    print(f"  Total lessons:        {len(lessons)}")
    print(f"  Potential mismatches: {len(mismatches)}")
    print(f"  Duplicate titles:     {dup_count}")
    print(f"  Missing fields:       {len(missing_fields)}")
    print("=" * 70)
    print("")
    
    # Write detailed report
    report_path = Path(__file__).parent.parent / "docs" / "CURRICULUM_AUDIT_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Curriculum Data Quality Audit Report\n\n")
        f.write(f"**Generated:** December 16, 2025\n")
        f.write(f"**Source:** lessons/365_day_calendar.json (Supabase sync)\n\n")
        f.write("---\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"| Metric | Count |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total lessons | {len(lessons)} |\n")
        f.write(f"| Potential mismatches | {len(mismatches)} |\n")
        f.write(f"| Duplicate titles | {dup_count} |\n")
        f.write(f"| Missing fields | {len(missing_fields)} |\n\n")
        
        if mismatches:
            f.write("---\n\n")
            f.write("## Potential Title/Objective Mismatches\n\n")
            f.write("These days have titles that don't obviously match their learning objectives:\n\n")
            f.write("| Day | Title | Learning Objective |\n")
            f.write("|-----|-------|-------------------|\n")
            for m in mismatches:
                obj_short = m['objective'][:60] + "..." if len(m['objective']) > 60 else m['objective']
                f.write(f"| {m['day']} | {m['title']} | {obj_short} |\n")
            f.write("\n")
        
        f.write("---\n\n")
        f.write("## Recommended Actions\n\n")
        if mismatches:
            f.write("1. Review each mismatch in Supabase `core_lessons` table\n")
            f.write("2. Determine if title or objective needs updating\n")
            f.write("3. After fixes, re-run sync scripts\n")
        else:
            f.write("No critical issues found. Curriculum is ready for use.\n")
    
    print(f"Detailed report written to: docs/CURRICULUM_AUDIT_REPORT.md")
    
    return mismatches

if __name__ == "__main__":
    audit_curriculum()
