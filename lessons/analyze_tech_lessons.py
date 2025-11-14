#!/usr/bin/env python3
"""
Analyze all technology lessons for value to humans in 2026 and beyond.
Focus on: relevance, timelessness, practical value, age-appropriateness.
"""

import json
import re
from collections import defaultdict

def load_calendar():
    """Load the 365-day calendar."""
    with open('365_day_calendar.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def is_tech_lesson(lesson):
    """Determine if a lesson is technology-focused."""
    title = lesson['title'].lower()
    objective = lesson.get('learning_objective', '').lower()
    text = title + " " + objective
    
    tech_keywords = [
        'technology', 'tech', 'digital', 'computer', 'internet', 'network',
        'software', 'hardware', 'ai', 'artificial intelligence', 'robot',
        'automation', 'innovation', 'device', 'machine', 'system',
        'algorithm', 'code', 'programming', 'data', 'cyber', 'cybersecurity',
        'virtual reality', 'augmented reality', 'blockchain', 'crypto',
        '3d printing', 'drone', 'sensor', 'iot', 'internet of things',
        'cloud computing', 'big data', 'machine learning', 'neural network',
        'quantum computing', 'biotechnology', 'nanotechnology', 'genetic engineering',
        'renewable energy', 'solar technology', 'wind technology', 'battery',
        'electric', 'electricity', 'communication technology', 'media',
        'social media', 'entertainment technology', 'educational technology',
        'medical technology', 'security technology', 'transportation technology',
        'manufacturing', 'industrial', 'automation', 'robotics'
    ]
    
    return any(keyword in text for keyword in tech_keywords)

def categorize_tech_lesson(lesson):
    """Categorize technology lesson by type."""
    title = lesson['title'].lower()
    objective = lesson.get('learning_objective', '').lower()
    text = title + " " + objective
    
    categories = {
        'foundational_computing': ['computer', 'programming', 'algorithm', 'code', 'software', 'hardware', 'binary'],
        'ai_ml': ['artificial intelligence', 'machine learning', 'neural network', 'ai', 'ml'],
        'internet_networks': ['internet', 'network', 'web', 'protocol', 'connectivity', 'bandwidth'],
        'digital_media': ['digital', 'media', 'social media', 'entertainment technology', 'streaming'],
        'emerging_tech': ['quantum', 'blockchain', 'crypto', 'virtual reality', 'augmented reality', 'ar', 'vr'],
        'biotech': ['biotechnology', 'genetic engineering', 'bio', 'genetic'],
        'energy_tech': ['renewable', 'solar technology', 'wind technology', 'battery', 'energy storage'],
        'robotics_automation': ['robot', 'robotics', 'automation', 'drone', 'autonomous'],
        'communication_tech': ['communication technology', 'telecommunication', 'wireless'],
        'medical_tech': ['medical technology', 'health technology', 'biomedical'],
        'educational_tech': ['educational technology', 'edtech', 'learning technology'],
        'security_tech': ['security technology', 'cybersecurity', 'cyber', 'privacy'],
        'manufacturing_tech': ['manufacturing', '3d printing', 'additive manufacturing', 'industrial'],
        'transportation_tech': ['transportation technology', 'autonomous vehicle', 'electric vehicle'],
        'innovation_process': ['innovation', 'invention', 'prototyping', 'design thinking', 'lean'],
        'general_tech': []  # Catch-all
    }
    
    matched = []
    for category, keywords in categories.items():
        if any(kw in text for kw in keywords):
            matched.append(category)
    
    return matched if matched else ['general_tech']

def assess_value_2026(lesson):
    """Assess value of tech lesson for 2026+."""
    title = lesson['title']
    objective = lesson.get('learning_objective', '')
    text = (title + " " + objective).lower()
    
    score = 0
    reasons = []
    
    # Positive factors (add points)
    
    # 1. Timeless principles (high value)
    timeless_keywords = ['principle', 'fundamental', 'foundation', 'understanding', 'how it works']
    if any(kw in text for kw in timeless_keywords):
        score += 3
        reasons.append("Teaches timeless principles")
    
    # 2. Critical thinking about tech (high value)
    critical_keywords = ['ethical', 'impact', 'implications', 'responsibility', 'governance', 
                        'democratic', 'human rights', 'equity', 'access', 'privacy']
    if any(kw in text for kw in critical_keywords):
        score += 4
        reasons.append("Addresses critical thinking about technology")
    
    # 3. Practical daily relevance (high value)
    practical_keywords = ['daily', 'everyday', 'personal', 'health', 'communication', 'learning']
    if any(kw in text for kw in practical_keywords):
        score += 2
        reasons.append("Practical daily relevance")
    
    # 4. Future-proof topics (high value)
    future_keywords = ['renewable', 'sustainable', 'climate', 'environment', 'energy', 
                      'quantum', 'ai', 'machine learning', 'biotechnology']
    if any(kw in text for kw in future_keywords):
        score += 3
        reasons.append("Future-relevant technology")
    
    # 5. Universal understanding (high value for age-less)
    universal_keywords = ['how', 'why', 'what', 'understanding', 'exploring', 'discovering']
    if any(kw in text for kw in universal_keywords):
        score += 1
        reasons.append("Universal understanding")
    
    # Negative factors (subtract points)
    
    # 1. Too specific/niche (low value)
    niche_keywords = ['patent', 'funding', 'venture capital', 'startup', 'ipo', 'market']
    if any(kw in text for kw in niche_keywords):
        score -= 2
        reasons.append("Too niche/specific")
    
    # 2. Business/economic focus (lower universal value)
    business_keywords = ['business', 'commerce', 'trade', 'economic', 'profit', 'revenue']
    if any(kw in text for kw in business_keywords):
        score -= 1
        reasons.append("Business-focused (less universal)")
    
    # 3. Rapidly changing tech (may be outdated)
    changing_keywords = ['social media', 'platform', 'app', 'software tool']
    if any(kw in text for kw in changing_keywords):
        score -= 1
        reasons.append("May become outdated quickly")
    
    # 4. Too abstract/process-focused (lower value)
    abstract_keywords = ['innovation process', 'methodology', 'framework', 'model', 'system']
    if len([kw for kw in abstract_keywords if kw in text]) > 2:
        score -= 2
        reasons.append("Too abstract/process-focused")
    
    # 5. Redundant with other lessons
    # (Will check separately)
    
    return score, reasons

def assess_age_less_value(lesson):
    """Assess if tech lesson is truly age-less."""
    title = lesson['title'].lower()
    objective = lesson.get('learning_objective', '').lower()
    text = title + " " + objective
    
    # Can a 3-year-old relate?
    child_keywords = ['how', 'why', 'what', 'amazing', 'wonderful', 'exploring', 'discovering']
    child_relatable = any(kw in text for kw in child_keywords)
    
    # Can an 80-year-old relate?
    elder_keywords = ['understanding', 'wisdom', 'experience', 'impact', 'meaning', 'value']
    elder_relatable = any(kw in text for kw in elder_keywords)
    
    # Is it observable/experiential?
    observable_keywords = ['see', 'feel', 'touch', 'hear', 'experience', 'observe', 'notice']
    observable = any(kw in text for kw in observable_keywords)
    
    # Is it about universal principles?
    universal_keywords = ['principle', 'fundamental', 'foundation', 'basic', 'core']
    universal = any(kw in text for kw in universal_keywords)
    
    score = 0
    if child_relatable: score += 1
    if elder_relatable: score += 1
    if observable: score += 1
    if universal: score += 1
    
    return score, {
        'child_relatable': child_relatable,
        'elder_relatable': elder_relatable,
        'observable': observable,
        'universal': universal
    }

def analyze_tech_lessons():
    """Comprehensive analysis of all technology lessons."""
    calendar = load_calendar()
    lessons = calendar['lessons']
    
    # Find all tech lessons
    tech_lessons = [l for l in lessons if is_tech_lesson(l)]
    
    print("=" * 80)
    print("TECHNOLOGY LESSONS ANALYSIS - Value for Humans in 2026+")
    print("=" * 80)
    print(f"\nTotal Technology Lessons Found: {len(tech_lessons)}")
    print(f"Percentage of Total: {len(tech_lessons)/len(lessons)*100:.1f}%")
    print()
    
    # Categorize
    print("1. TECHNOLOGY LESSON CATEGORIES")
    print("-" * 80)
    
    category_lessons = defaultdict(list)
    for lesson in tech_lessons:
        categories = categorize_tech_lesson(lesson)
        for cat in categories:
            category_lessons[cat].append(lesson)
    
    for category, lesson_list in sorted(category_lessons.items(), 
                                       key=lambda x: len(x[1]), reverse=True):
        print(f"{category:25s}: {len(lesson_list):3d} lessons")
    
    # Assess value
    print("\n2. VALUE ASSESSMENT FOR 2026+")
    print("-" * 80)
    
    high_value = []
    medium_value = []
    low_value = []
    
    for lesson in tech_lessons:
        value_score, reasons = assess_value_2026(lesson)
        age_score, age_factors = assess_age_less_value(lesson)
        
        lesson['value_score'] = value_score
        lesson['age_score'] = age_score
        lesson['value_reasons'] = reasons
        lesson['age_factors'] = age_factors
        
        total_score = value_score + (age_score * 0.5)  # Age-less is important but secondary
        
        if total_score >= 4:
            high_value.append(lesson)
        elif total_score >= 1:
            medium_value.append(lesson)
        else:
            low_value.append(lesson)
    
    print(f"High Value (Score ≥4):     {len(high_value):3d} lessons")
    print(f"Medium Value (Score 1-3):  {len(medium_value):3d} lessons")
    print(f"Low Value (Score <1):      {len(low_value):3d} lessons")
    
    # Show high value lessons
    print("\n3. HIGH VALUE LESSONS (Keep/Expand)")
    print("-" * 80)
    for lesson in sorted(high_value, key=lambda x: x['value_score'], reverse=True)[:20]:
        print(f"Day {lesson['day']:3d}: {lesson['title']}")
        print(f"         Score: {lesson['value_score']:.1f} | Age-less: {lesson['age_score']}/4")
        print(f"         Reasons: {', '.join(lesson['value_reasons'][:3])}")
        print()
    
    # Show low value lessons
    print("\n4. LOW VALUE LESSONS (Consider Replacing)")
    print("-" * 80)
    for lesson in sorted(low_value, key=lambda x: x['value_score'])[:20]:
        print(f"Day {lesson['day']:3d}: {lesson['title']}")
        print(f"         Score: {lesson['value_score']:.1f} | Age-less: {lesson['age_score']}/4")
        print(f"         Issues: {', '.join(lesson['value_reasons'][:3])}")
        print()
    
    # Redundancy check
    print("\n5. REDUNDANT/SIMILAR TECH LESSONS")
    print("-" * 80)
    
    title_words = defaultdict(list)
    for lesson in tech_lessons:
        words = set(re.findall(r'\b\w{4,}\b', lesson['title'].lower()))
        for word in words:
            if word not in ['technology', 'innovation', 'understanding', 'exploring']:
                title_words[word].append(lesson)
    
    redundant = {word: lessons for word, lessons in title_words.items() 
                if len(lessons) > 2 and len(word) > 4}
    
    if redundant:
        print(f"Found {len(redundant)} topics with 3+ lessons:")
        for word, lesson_list in sorted(redundant.items(), 
                                       key=lambda x: len(x[1]), reverse=True)[:10]:
            print(f"\n'{word}' appears in {len(lesson_list)} lessons:")
            for lesson in lesson_list[:5]:
                print(f"   Day {lesson['day']:3d}: {lesson['title']}")
    
    # Recommendations
    print("\n6. RECOMMENDATIONS")
    print("-" * 80)
    
    recommendations = []
    
    if len(low_value) > 30:
        recommendations.append(f"Replace {len(low_value)} low-value tech lessons with universal topics")
    
    if len(high_value) < 50:
        recommendations.append(f"Only {len(high_value)} high-value tech lessons - focus on expanding these")
    
    # Check for too many innovation process lessons
    innovation_process = category_lessons.get('innovation_process', [])
    if len(innovation_process) > 15:
        recommendations.append(f"Too many innovation process lessons ({len(innovation_process)}) - consolidate")
    
    # Check for missing critical topics
    critical_topics = ['privacy', 'security', 'ethics', 'equity', 'access', 'sustainability']
    missing_critical = []
    for topic in critical_topics:
        found = any(topic in (l['title'] + l.get('learning_objective', '')).lower() 
                   for l in tech_lessons)
        if not found:
            missing_critical.append(topic)
    
    if missing_critical:
        recommendations.append(f"Missing critical tech topics: {', '.join(missing_critical)}")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    if not recommendations:
        print("✅ Tech lessons appear well-balanced")
    
    # Summary statistics
    print("\n7. SUMMARY STATISTICS")
    print("-" * 80)
    print(f"Total tech lessons: {len(tech_lessons)}")
    print(f"High value: {len(high_value)} ({len(high_value)/len(tech_lessons)*100:.1f}%)")
    print(f"Medium value: {len(medium_value)} ({len(medium_value)/len(tech_lessons)*100:.1f}%)")
    print(f"Low value: {len(low_value)} ({len(low_value)/len(tech_lessons)*100:.1f}%)")
    print(f"Average value score: {sum(l['value_score'] for l in tech_lessons)/len(tech_lessons):.2f}")
    print(f"Average age-less score: {sum(l['age_score'] for l in tech_lessons)/len(tech_lessons):.2f}/4")
    
    return {
        'tech_lessons': tech_lessons,
        'high_value': high_value,
        'medium_value': medium_value,
        'low_value': low_value,
        'category_lessons': category_lessons,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    results = analyze_tech_lessons()
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

