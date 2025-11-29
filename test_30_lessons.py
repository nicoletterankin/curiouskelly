#!/usr/bin/env python3
"""
Test script to verify all 30 lessons load correctly from Supabase
"""

import psycopg2
import json
from collections import defaultdict

# Database connection
DB_URL = "postgresql://antigravity:antigravity123@localhost:5432/antigravity_dev"

# Expected phases
EXPECTED_PHASES = ['welcome', 'q1', 'q2', 'q3', 'wisdom']

# Archetypes to test
ARCHETYPES = ['Sage', 'Jester', 'Ruler']  # The 3 used by tone mapping

def test_lessons():
    """Test all 30 lessons"""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    print("\n" + "="*70)
    print("TESTING 30 LESSONS - SUPABASE DATA VERIFICATION")
    print("="*70 + "\n")
    
    results = {
        'passed': [],
        'failed': [],
        'warnings': []
    }
    
    for day in range(1, 31):
        print(f"📅 Day {day}:", end=" ")
        
        try:
            # Get core lesson
            cur.execute("""
                SELECT id, topic, universal_truth
                FROM core_lessons
                WHERE day_number = %s
            """, (day,))
            
            core_lesson = cur.fetchone()
            
            if not core_lesson:
                print(f"❌ FAIL - No core lesson found")
                results['failed'].append({
                    'day': day,
                    'error': 'No core lesson in database'
                })
                continue
            
            lesson_id, topic, truth = core_lesson
            print(f"{topic}", end=" ")
            
            # Test each archetype
            archetype_results = {}
            
            for archetype in ARCHETYPES:
                cur.execute("""
                    SELECT phase, content
                    FROM lesson_atoms
                    WHERE core_lesson_id = %s
                    AND archetype = %s
                    ORDER BY phase
                """, (lesson_id, archetype))
                
                atoms = cur.fetchall()
                
                if not atoms:
                    archetype_results[archetype] = {
                        'status': 'missing',
                        'phases': []
                    }
                    continue
                
                # Check phases
                phases_found = [atom[0] for atom in atoms]
                missing_phases = set(EXPECTED_PHASES) - set(phases_found)
                
                # Validate content
                issues = []
                for phase, content_json in atoms:
                    content = content_json
                    
                    # Check script exists and is not empty
                    script = content.get('script') or content.get('text', '')
                    if not script or len(script.strip()) < 10:
                        issues.append(f"{phase}: script too short or missing")
                    
                    # Check question phases have options
                    if phase in ['q1', 'q2', 'q3']:
                        options = content.get('options', [])
                        if not options or len(options) < 2:
                            issues.append(f"{phase}: missing or insufficient options")
                
                archetype_results[archetype] = {
                    'status': 'ok' if not missing_phases and not issues else 'warning',
                    'phases': phases_found,
                    'missing_phases': list(missing_phases),
                    'issues': issues
                }
            
            # Determine overall status
            all_ok = all(r['status'] == 'ok' for r in archetype_results.values())
            any_missing = any(r['status'] == 'missing' for r in archetype_results.values())
            
            if any_missing:
                print(f"❌ FAIL - Missing archetypes")
                results['failed'].append({
                    'day': day,
                    'topic': topic,
                    'error': 'Missing archetype data',
                    'details': archetype_results
                })
            elif all_ok:
                print(f"✅ PASS")
                results['passed'].append({
                    'day': day,
                    'topic': topic
                })
            else:
                print(f"⚠️  WARN - Has issues")
                results['warnings'].append({
                    'day': day,
                    'topic': topic,
                    'details': archetype_results
                })
        
        except Exception as e:
            print(f"❌ ERROR - {e}")
            results['failed'].append({
                'day': day,
                'error': str(e)
            })
    
    conn.close()
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✅ PASSED:  {len(results['passed'])}/30")
    print(f"⚠️  WARNINGS: {len(results['warnings'])}/30")
    print(f"❌ FAILED:  {len(results['failed'])}/30")
    print("="*70 + "\n")
    
    # Show warnings details
    if results['warnings']:
        print("⚠️  WARNINGS DETAILS:")
        for item in results['warnings']:
            print(f"\n  Day {item['day']}: {item['topic']}")
            for archetype, details in item['details'].items():
                if details['status'] != 'ok':
                    print(f"    {archetype}:")
                    if details.get('missing_phases'):
                        print(f"      Missing phases: {', '.join(details['missing_phases'])}")
                    if details.get('issues'):
                        for issue in details['issues']:
                            print(f"      Issue: {issue}")
        print()
    
    # Show failures details
    if results['failed']:
        print("❌ FAILURES DETAILS:")
        for item in results['failed']:
            print(f"\n  Day {item['day']}: {item.get('topic', 'Unknown')}")
            print(f"    Error: {item['error']}")
        print()
    
    # Overall result
    if len(results['passed']) == 30:
        print("🎉 ALL 30 LESSONS PASSED! Ready for launch!")
        return True
    elif len(results['failed']) == 0:
        print("✅ All lessons present with minor warnings. Should work fine.")
        return True
    else:
        print("⚠️  Some lessons have issues. Review failures above.")
        return False


def test_archetype_coverage():
    """Test that all 3 tone-mapped archetypes are present"""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    print("\n" + "="*70)
    print("ARCHETYPE COVERAGE TEST")
    print("="*70 + "\n")
    
    for archetype in ARCHETYPES:
        cur.execute("""
            SELECT COUNT(DISTINCT day_number)
            FROM lesson_atoms
            WHERE archetype = %s
            AND day_number BETWEEN 1 AND 30
        """, (archetype,))
        
        count = cur.fetchone()[0]
        status = "✅" if count == 30 else "❌"
        print(f"{status} {archetype}: {count}/30 days")
    
    conn.close()
    print()


def test_content_quality():
    """Test content quality metrics"""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    print("\n" + "="*70)
    print("CONTENT QUALITY METRICS")
    print("="*70 + "\n")
    
    # Average script length
    cur.execute("""
        SELECT 
            phase,
            AVG(LENGTH(content->>'script')) as avg_length,
            MIN(LENGTH(content->>'script')) as min_length,
            MAX(LENGTH(content->>'script')) as max_length
        FROM lesson_atoms
        WHERE day_number BETWEEN 1 AND 30
        AND archetype IN ('Sage', 'Jester', 'Ruler')
        GROUP BY phase
        ORDER BY phase
    """)
    
    print("Script Lengths by Phase:")
    for row in cur.fetchall():
        phase, avg_len, min_len, max_len = row
        print(f"  {phase:8s}: avg={avg_len:5.0f} chars  (min={min_len}, max={max_len})")
    
    # Options count for question phases
    cur.execute("""
        SELECT 
            phase,
            COUNT(*) as total,
            SUM(CASE WHEN jsonb_array_length(content->'options') >= 3 THEN 1 ELSE 0 END) as with_3_options
        FROM lesson_atoms
        WHERE day_number BETWEEN 1 AND 30
        AND phase IN ('q1', 'q2', 'q3')
        AND archetype IN ('Sage', 'Jester', 'Ruler')
        GROUP BY phase
        ORDER BY phase
    """)
    
    print("\nQuestion Phases with Options:")
    for row in cur.fetchall():
        phase, total, with_options = row
        pct = (with_options / total * 100) if total > 0 else 0
        print(f"  {phase}: {with_options}/{total} have 3+ options ({pct:.0f}%)")
    
    conn.close()
    print()


if __name__ == "__main__":
    print("\n🧪 CURIOUS KELLY - 30 LESSON TEST SUITE")
    print("Testing days 1-30 with Sage, Jester, Ruler archetypes\n")
    
    # Run tests
    test_archetype_coverage()
    test_content_quality()
    success = test_lessons()
    
    # Exit code
    exit(0 if success else 1)


