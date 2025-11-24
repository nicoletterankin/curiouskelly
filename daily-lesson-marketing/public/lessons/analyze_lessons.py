#!/usr/bin/env python3
"""
Comprehensive analysis of 365-day lesson calendar.
Analyzes duplicates, gaps, categorization, and overall quality.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
import re

def load_calendar():
    """Load the 365-day calendar."""
    with open('365_day_calendar.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_title(title):
    """Normalize title for comparison."""
    # Remove common words and normalize
    title = title.lower()
    # Remove common prefixes/suffixes
    title = re.sub(r'^(the|a|an|our|your|amazing|magnificent|incredible|wonderful)\s+', '', title)
    title = re.sub(r'\s*-\s*.*$', '', title)  # Remove subtitle after dash
    title = re.sub(r'[^\w\s]', '', title)  # Remove punctuation
    return title.strip()

def extract_keywords(title):
    """Extract key topic words from title."""
    # Remove common words
    stop_words = {'the', 'a', 'an', 'our', 'your', 'amazing', 'magnificent', 
                  'incredible', 'wonderful', 'how', 'what', 'why', 'when', 
                  'where', 'understanding', 'exploring', 'learning', 'about',
                  'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with',
                  'from', 'through', 'that', 'this', 'these', 'those'}
    
    words = re.findall(r'\b\w+\b', title.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return set(keywords)

def categorize_lesson(title, objective):
    """Categorize lesson into broad categories."""
    text = (title + " " + objective).lower()
    
    categories = {
        'science': ['physics', 'chemistry', 'biology', 'astronomy', 'geology', 
                   'molecular', 'atomic', 'quantum', 'nuclear', 'energy', 
                   'force', 'wave', 'light', 'sound', 'matter', 'particle',
                   'solar', 'moon', 'earth', 'ocean', 'atmosphere', 'climate',
                   'evolution', 'genetics', 'cell', 'organism', 'ecosystem'],
        'technology': ['computer', 'digital', 'internet', 'network', 'software',
                      'hardware', 'ai', 'robot', 'automation', 'innovation',
                      'technology', 'tech', 'device', 'machine', 'system',
                      'algorithm', 'code', 'programming', 'data', 'cyber'],
        'mathematics': ['math', 'mathematical', 'number', 'equation', 'formula',
                       'calculation', 'geometry', 'algebra', 'calculus',
                       'statistics', 'probability', 'fractal', 'pattern',
                       'graph', 'measurement', 'quantity'],
        'arts': ['art', 'painting', 'drawing', 'sculpture', 'music', 'dance',
                'theater', 'performance', 'creative', 'aesthetic', 'design',
                'poetry', 'literature', 'writing', 'story', 'narrative',
                'photography', 'film', 'visual', 'expression'],
        'social_sciences': ['society', 'culture', 'social', 'human', 'behavior',
                           'psychology', 'sociology', 'anthropology', 'history',
                           'politics', 'government', 'democracy', 'rights',
                           'economics', 'trade', 'business', 'commerce'],
        'health_wellness': ['health', 'wellness', 'fitness', 'exercise', 'nutrition',
                           'medical', 'medicine', 'disease', 'healing', 'body',
                           'mind', 'mental', 'emotional', 'stress', 'sleep',
                           'aging', 'longevity', 'prevention'],
        'skills_personal': ['skill', 'learning', 'thinking', 'communication',
                          'leadership', 'teamwork', 'problem', 'decision',
                          'creativity', 'innovation', 'productivity', 'habit',
                          'goal', 'time', 'management', 'focus', 'memory'],
        'nature_environment': ['nature', 'environment', 'conservation', 'sustainability',
                             'ecosystem', 'biodiversity', 'wildlife', 'plant',
                             'animal', 'forest', 'ocean', 'water', 'air',
                             'pollution', 'renewable', 'green'],
        'philosophy_ethics': ['philosophy', 'ethics', 'moral', 'wisdom', 'meaning',
                             'truth', 'knowledge', 'belief', 'spiritual',
                             'religion', 'meditation', 'mindfulness'],
        'history_culture': ['history', 'civilization', 'ancient', 'culture',
                           'tradition', 'heritage', 'exploration', 'discovery',
                           'war', 'peace', 'conflict', 'movement']
    }
    
    matched = []
    for category, keywords in categories.items():
        if any(kw in text for kw in keywords):
            matched.append(category)
    
    return matched if matched else ['general']

def analyze_lessons():
    """Perform comprehensive analysis."""
    calendar = load_calendar()
    lessons = calendar['lessons']
    
    print("=" * 80)
    print("365-DAY LESSON CALENDAR ANALYSIS")
    print("=" * 80)
    print()
    
    # 1. Duplicate Analysis
    print("1. DUPLICATE ANALYSIS")
    print("-" * 80)
    
    title_counts = Counter([l['title'] for l in lessons])
    duplicates = {title: count for title, count in title_counts.items() if count > 1}
    
    if duplicates:
        print(f"⚠️  Found {len(duplicates)} exact duplicate titles:")
        for title, count in duplicates.items():
            print(f"   - '{title}' appears {count} times")
    else:
        print("✅ No exact duplicate titles found")
    
    # Similar titles (normalized)
    normalized_titles = {}
    for lesson in lessons:
        norm = normalize_title(lesson['title'])
        if norm not in normalized_titles:
            normalized_titles[norm] = []
        normalized_titles[norm].append(lesson)
    
    similar = {norm: lessons for norm, lessons in normalized_titles.items() 
               if len(lessons) > 1 and len(norm) > 10}
    
    if similar:
        print(f"\n⚠️  Found {len(similar)} groups of similar titles:")
        for norm, lesson_list in list(similar.items())[:10]:
            if len(lesson_list) > 1:
                print(f"   Similar: '{norm}'")
                for lesson in lesson_list:
                    print(f"      - Day {lesson['day']}: {lesson['title']}")
    
    # 2. Keyword Overlap Analysis
    print("\n2. KEYWORD OVERLAP ANALYSIS")
    print("-" * 80)
    
    keyword_lessons = defaultdict(list)
    for lesson in lessons:
        keywords = extract_keywords(lesson['title'])
        for keyword in keywords:
            keyword_lessons[keyword].append(lesson)
    
    # Find keywords that appear in many lessons
    frequent_keywords = {kw: lessons for kw, lessons in keyword_lessons.items() 
                        if len(lessons) > 3 and len(kw) > 4}
    
    print(f"Most common topic keywords (appearing in 4+ lessons):")
    for keyword, lesson_list in sorted(frequent_keywords.items(), 
                                      key=lambda x: len(x[1]), reverse=True)[:20]:
        print(f"   '{keyword}': {len(lesson_list)} lessons")
    
    # 3. Category Distribution
    print("\n3. CATEGORY DISTRIBUTION")
    print("-" * 80)
    
    category_counts = defaultdict(int)
    for lesson in lessons:
        categories = categorize_lesson(lesson['title'], lesson.get('learning_objective', ''))
        for cat in categories:
            category_counts[cat] += 1
    
    print("Lessons by category:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(lessons)) * 100
        print(f"   {category:20s}: {count:3d} lessons ({percentage:5.1f}%)")
    
    # 4. DNA Lesson Analysis
    print("\n4. DNA LESSON ANALYSIS")
    print("-" * 80)
    
    dna_lessons = [l for l in lessons if l.get('has_dna', False)]
    non_dna_lessons = [l for l in lessons if not l.get('has_dna', False)]
    
    print(f"DNA lessons: {len(dna_lessons)} ({len(dna_lessons)/len(lessons)*100:.1f}%)")
    print(f"Non-DNA lessons: {len(non_dna_lessons)} ({len(non_dna_lessons)/len(lessons)*100:.1f}%)")
    
    # DNA lessons by category
    dna_categories = defaultdict(int)
    for lesson in dna_lessons:
        categories = categorize_lesson(lesson['title'], lesson.get('learning_objective', ''))
        for cat in categories:
            dna_categories[cat] += 1
    
    print("\nDNA lessons by category:")
    for category, count in sorted(dna_categories.items(), key=lambda x: x[1], reverse=True):
        print(f"   {category:20s}: {count:2d} lessons")
    
    # 5. Gap Analysis - Missing Topics
    print("\n5. GAP ANALYSIS - POTENTIALLY MISSING TOPICS")
    print("-" * 80)
    
    all_keywords = set()
    for lesson in lessons:
        all_keywords.update(extract_keywords(lesson['title']))
    
    # Common universal topics that might be missing
    potential_topics = {
        'friendship', 'kindness', 'empathy', 'gratitude', 'patience',
        'honesty', 'integrity', 'respect', 'compassion', 'forgiveness',
        'curiosity', 'wonder', 'imagination', 'play', 'laughter',
        'seasons', 'weather', 'clouds', 'rain', 'snow', 'wind',
        'trees', 'flowers', 'birds', 'insects', 'butterflies',
        'shadows', 'reflections', 'mirrors', 'colors', 'rainbow',
        'time', 'clocks', 'calendars', 'yesterday', 'tomorrow',
        'family', 'home', 'community', 'neighborhood', 'school',
        'sharing', 'cooperation', 'helping', 'caring', 'listening',
        'growth', 'change', 'transformation', 'cycles', 'patterns',
        'balance', 'harmony', 'peace', 'calm', 'stillness'
    }
    
    missing = potential_topics - all_keywords
    if missing:
        print(f"Potentially missing universal topics ({len(missing)}):")
        for topic in sorted(missing):
            print(f"   - {topic}")
    else:
        print("✅ All common universal topics appear to be covered")
    
    # 6. Age-Appropriateness Check
    print("\n6. AGE-APPROPRIATENESS CHECK")
    print("-" * 80)
    
    # Look for topics that might be too specific to certain ages
    age_specific_keywords = {
        'retirement', 'pension', '401k', 'mortgage', 'taxes',
        'dating', 'romance', 'marriage', 'divorce',
        'homework', 'exam', 'test', 'grade', 'college',
        'job', 'career', 'salary', 'promotion', 'resume',
        'pregnancy', 'childbirth', 'menopause'
    }
    
    age_specific_lessons = []
    for lesson in lessons:
        text = (lesson['title'] + " " + lesson.get('learning_objective', '')).lower()
        if any(kw in text for kw in age_specific_keywords):
            age_specific_lessons.append(lesson)
    
    if age_specific_lessons:
        print(f"⚠️  Found {len(age_specific_lessons)} lessons that might be age-specific:")
        for lesson in age_specific_lessons[:10]:
            print(f"   Day {lesson['day']}: {lesson['title']}")
    else:
        print("✅ No obviously age-specific topics found")
    
    # 7. Month Distribution
    print("\n7. MONTH DISTRIBUTION")
    print("-" * 80)
    
    month_counts = defaultdict(int)
    for lesson in lessons:
        month = lesson['date'].split()[0] if ' ' in lesson['date'] else 'Unknown'
        month_counts[month] += 1
    
    print("Lessons per month:")
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    for month in months:
        count = month_counts.get(month, 0)
        print(f"   {month:12s}: {count:3d} lessons")
    
    # 8. Recommendations
    print("\n8. RECOMMENDATIONS")
    print("-" * 80)
    
    recommendations = []
    
    # Check for balance
    if category_counts['science'] > 100:
        recommendations.append("Consider reducing science lessons or balancing with other categories")
    
    if category_counts['technology'] < 20:
        recommendations.append("Consider adding more technology/innovation topics")
    
    if category_counts['arts'] < 30:
        recommendations.append("Consider adding more arts/creative expression topics")
    
    if category_counts['health_wellness'] < 30:
        recommendations.append("Consider adding more health/wellness topics")
    
    if len(dna_lessons) < 50:
        recommendations.append(f"Only {len(dna_lessons)} DNA lessons - consider creating more DNA files")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print("   ✅ Curriculum appears well-balanced")
    
    # 9. Quality Check - Lessons with very short/long titles
    print("\n9. QUALITY CHECKS")
    print("-" * 80)
    
    short_titles = [l for l in lessons if len(l['title']) < 10]
    long_titles = [l for l in lessons if len(l['title']) > 80]
    
    if short_titles:
        print(f"⚠️  {len(short_titles)} lessons with very short titles (<10 chars):")
        for lesson in short_titles[:5]:
            print(f"   Day {lesson['day']}: '{lesson['title']}'")
    
    if long_titles:
        print(f"⚠️  {len(long_titles)} lessons with very long titles (>80 chars):")
        for lesson in long_titles[:5]:
            print(f"   Day {lesson['day']}: '{lesson['title']}'")
    
    if not short_titles and not long_titles:
        print("✅ Title lengths appear appropriate")
    
    # 10. Summary Statistics
    print("\n10. SUMMARY STATISTICS")
    print("-" * 80)
    print(f"Total lessons: {len(lessons)}")
    print(f"DNA lessons: {len(dna_lessons)}")
    print(f"Unique titles: {len(set(l['title'] for l in lessons))}")
    print(f"Categories covered: {len(category_counts)}")
    print(f"Average title length: {sum(len(l['title']) for l in lessons) / len(lessons):.1f} characters")
    
    return {
        'lessons': lessons,
        'duplicates': duplicates,
        'similar': similar,
        'category_counts': category_counts,
        'dna_lessons': dna_lessons,
        'missing_topics': missing,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    results = analyze_lessons()
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

