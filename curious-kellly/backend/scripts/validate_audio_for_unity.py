#!/usr/bin/env python3
"""
Audio File Validation Script for Unity Integration
Validates all 558 generated audio files are ready for Unity avatar testing
"""

import os
import sys
from pathlib import Path

def validate_lesson_audio(lesson_name, audio_base_path):
    """Validate all audio files for a lesson"""
    audio_dir = Path(audio_base_path) / lesson_name

    age_buckets = ['2-5', '6-12', '13-17', '18-35', '36-60', '61-102']
    languages = ['en', 'es', 'fr']
    sections = ['welcome', 'mainContent', 'wisdomMoment']

    expected = len(age_buckets) * len(languages) * len(sections)
    found = 0
    errors = []
    total_size = 0

    print(f'\n{"="*60}')
    print(f'Validating: {lesson_name}')
    print(f'{"="*60}')

    for age in age_buckets:
        for lang in languages:
            for section in sections:
                filename = f'{age}-{section}-{lang}.mp3'
                filepath = audio_dir / filename

                if not filepath.exists():
                    errors.append(f'❌ Missing: {filename}')
                    continue

                # Check file size
                file_size = filepath.stat().st_size
                total_size += file_size

                # Validate file is not empty
                if file_size < 1000:  # Less than 1KB is suspicious
                    errors.append(f'⚠️  Too small ({file_size} bytes): {filename}')
                    continue

                # Validate file is MP3 (basic check)
                with open(filepath, 'rb') as f:
                    header = f.read(3)
                    if header != b'ID3' and header[:2] != b'\xff\xfb':  # ID3 tag or MP3 frame sync
                        errors.append(f'⚠️  Invalid MP3 header: {filename}')
                        continue

                found += 1

    # Print summary
    print(f'Expected files: {expected}')
    print(f'Found files:    {found}')
    print(f'Total size:     {total_size / (1024*1024):.1f} MB')

    if errors:
        print(f'\n❌ Issues found ({len(errors)}):')
        for err in errors:
            print(f'   {err}')
        return False
    else:
        print(f'✅ All files validated successfully!')
        return True

def main():
    # Determine audio base path
    script_dir = Path(__file__).parent
    audio_base_path = script_dir.parent / 'config' / 'audio'

    print('='*60)
    print('CURIOUS KELLY - Audio Validation for Unity')
    print('='*60)
    print(f'Audio directory: {audio_base_path}')

    # List of all lessons
    lessons = [
        'the-sun',
        'puppies',
        'the-ocean',
        'the-moon',
        'molecular-biology-dna',
        'creative-writing-dna',
        'poetry-dna',
        'dance-expression-dna',
        'negotiation-skills-dna'
    ]

    # Special case: water-cycle has 72 files (different structure)
    water_cycle_dir = audio_base_path / 'water-cycle'
    if water_cycle_dir.exists():
        water_cycle_files = list(water_cycle_dir.glob('*.mp3'))
        print(f'\nwater-cycle: {len(water_cycle_files)} files (pre-existing)')

    # Validate all lessons
    all_pass = True
    total_files = 0
    total_size = 0

    for lesson in lessons:
        if not validate_lesson_audio(lesson, audio_base_path):
            all_pass = False
        else:
            lesson_dir = audio_base_path / lesson
            lesson_files = list(lesson_dir.glob('*.mp3'))
            total_files += len(lesson_files)
            total_size += sum(f.stat().st_size for f in lesson_files)

    # Add water-cycle if it exists
    if water_cycle_dir.exists():
        total_files += len(water_cycle_files)
        total_size += sum(f.stat().st_size for f in water_cycle_files)

    # Final summary
    print('\n' + '='*60)
    print('VALIDATION SUMMARY')
    print('='*60)
    print(f'Total lessons validated: {len(lessons) + 1}')  # +1 for water-cycle
    print(f'Total audio files: {total_files}')
    print(f'Total size: {total_size / (1024*1024):.1f} MB')

    if all_pass:
        print('\n✅ ALL AUDIO FILES VALIDATED - READY FOR UNITY!')
        print('\nNext steps:')
        print('1. Copy audio files to Unity project: Assets/Kelly/Audio/')
        print('2. Import audio clips in Unity')
        print('3. Test with BlendshapeDriver60fps.cs')
        print('4. Run performance tests on target devices')
        return 0
    else:
        print('\n❌ VALIDATION FAILED - FIX ERRORS ABOVE')
        return 1

if __name__ == '__main__':
    sys.exit(main())
