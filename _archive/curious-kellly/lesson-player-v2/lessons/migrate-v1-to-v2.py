#!/usr/bin/env python3
"""
V1 to DNA v2 Lesson Migration Script
Converts old schema lesson files to new PhaseDNA v2 format
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

def migrate_lesson_v1_to_v2(input_file):
    """
    Migrate a V1 schema lesson to DNA v2 format
    """
    print(f"🔄 Migrating: {input_file}")
    
    # Read V1 file
    with open(input_file, 'r', encoding='utf-8') as f:
        v1_data = json.load(f)
    
    # Extract lesson ID and title
    lesson_id = v1_data.get('id', Path(input_file).stem)
    title = v1_data.get('title', 'Migrated Lesson')
    
    print(f"   Title: {title}")
    print(f"   ID: {lesson_id}")
    
    # Create DNA v2 structure
    v2_data = {
        "id": lesson_id,
        "title": f"Migrated: {title}",
        "version": "2.0.0",
        "createdAt": datetime.utcnow().isoformat() + "Z",
        "updatedAt": datetime.utcnow().isoformat() + "Z",
        "author": "V1 to V2 Migration Script",
        "description": v1_data.get('description', ''),
        
        # Metadata from V1
        "calendar": v1_data.get('calendar', {}),
        
        # NEW: Universal concept (placeholder - needs manual enrichment)
        "universal_concept": "requires_manual_enrichment",
        "universal_concept_translations": {
            "en": "Requires manual enrichment",
            "es": "Requiere enriquecimiento manual",
            "fr": "Nécessite un enrichissement manuel"
        },
        
        # NEW: Core principle (placeholder)
        "core_principle": "requires_manual_enrichment",
        "core_principle_translations": {
            "en": "Requires manual enrichment",
            "es": "Requiere enriquecimiento manual",
            "fr": "Nécessite un enrichissement manuel"
        },
        
        # NEW: Learning essence
        "learning_essence": v1_data.get('description', 'Requires manual enrichment'),
        "learning_essence_translations": {
            "en": v1_data.get('description', 'Requires manual enrichment'),
            "es": "Requiere traducción manual",
            "fr": "Nécessite une traduction manuelle"
        },
        
        "metadata": v1_data.get('metadata', {
            "category": "general",
            "difficulty": "beginner",
            "duration": {"min": 5, "max": 13},
            "tags": [],
            "prerequisites": [],
            "learningOutcomes": []
        }),
        
        "ageVariants": {}
    }
    
    # Migrate age variants
    age_map = {
        "2-5": {"kellyAge": 3, "persona": "playful-toddler"},
        "6-12": {"kellyAge": 9, "persona": "curious-kid"},
        "13-17": {"kellyAge": 15, "persona": "enthusiastic-teen"},
        "18-35": {"kellyAge": 27, "persona": "knowledgeable-adult"},
        "36-60": {"kellyAge": 48, "persona": "wise-mentor"},
        "61-102": {"kellyAge": 82, "persona": "reflective-elder"}
    }
    
    v1_age_variants = v1_data.get('ageVariants', {})
    
    for age_key, age_config in age_map.items():
        if age_key not in v1_age_variants:
            print(f"   ⚠️  Missing age variant: {age_key}")
            continue
        
        v1_variant = v1_age_variants[age_key]
        
        # Create DNA v2 age variant structure
        v2_variant = {
            "title": v1_variant.get('title', title),
            "description": v1_variant.get('description', ''),
            "video": v1_variant.get('video', f"kelly_{lesson_id}_{age_key}.mp4"),
            "script": v1_variant.get('script', ''),
            "kellyAge": age_config["kellyAge"],
            "kellyPersona": age_config["persona"],
            
            # Voice profile
            "voiceProfile": {
                "provider": "elevenlabs",
                "voiceId": "wAdymQH5YucAkXwmrdL0",
                "speechRate": 1.0 if age_key == "6-12" else 0.85 if age_key == "2-5" else 1.0,
                "pitch": 0,
                "energy": "bright",
                "language": "en-US"
            },
            
            # NEW: Core metaphor (placeholder)
            "core_metaphor": "requires_manual_enrichment",
            "core_metaphor_translations": {
                "en": "Requires manual enrichment",
                "es": "Requiere enriquecimiento manual",
                "fr": "Nécessite un enrichissement manuel"
            },
            
            # NEW: Attention span
            "attention_span": "3-4_minutes" if age_key == "2-5" else "5-6_minutes",
            "cognitive_focus": "requires_manual_enrichment",
            
            # Examples (from V1 if exists)
            "examples": v1_variant.get('examples', []),
            
            # NEW: Language structure (multilingual)
            "language": {
                "en": {
                    "title": v1_variant.get('title', title),
                    "welcome": v1_variant.get('script', ''),
                    "mainContent": v1_variant.get('description', ''),
                    "keyPoints": v1_variant.get('objectives', []),
                    "interactionPrompts": [
                        "What do you think about this?",
                        "Can you share your thoughts?"
                    ],
                    "wisdomMoment": "Wonderful!",
                    "core_metaphor": "Requires manual enrichment",
                    "abstract_concepts": {},
                    "cta": "Keep exploring!",
                    "summary": "Great learning today!"
                },
                "es": {
                    "title": "Requiere traducción manual",
                    "welcome": "¡Requiere traducción manual!",
                    "mainContent": "Requiere traducción manual",
                    "keyPoints": [],
                    "interactionPrompts": ["¿Qué piensas sobre esto?"],
                    "wisdomMoment": "¡Maravilloso!",
                    "core_metaphor": "Requiere traducción manual",
                    "abstract_concepts": {},
                    "cta": "¡Sigue explorando!",
                    "summary": "¡Gran aprendizaje hoy!"
                },
                "fr": {
                    "title": "Nécessite une traduction manuelle",
                    "welcome": "Nécessite une traduction manuelle!",
                    "mainContent": "Nécessite une traduction manuelle",
                    "keyPoints": [],
                    "interactionPrompts": ["Qu'en penses-tu?"],
                    "wisdomMoment": "Merveilleux!",
                    "core_metaphor": "Nécessite une traduction manuelle",
                    "abstract_concepts": {},
                    "cta": "Continuez à explorer!",
                    "summary": "Excellent apprentissage aujourd'hui!"
                }
            },
            
            # From V1
            "objectives": v1_variant.get('objectives', []),
            "vocabulary": v1_variant.get('vocabulary', {
                "keyTerms": [],
                "complexity": "simple",
                "explanations": {}
            }),
            
            # NEW: Abstract concepts (placeholder)
            "abstract_concepts": {},
            "abstract_concepts_translations": {},
            
            # NEW: Pacing
            "pacing": {
                "speechRate": "slow" if age_key == "2-5" else "moderate",
                "pauseFrequency": "frequent" if age_key == "2-5" else "moderate",
                "interactionLevel": "high" if age_key in ["2-5", "6-12"] else "moderate"
            },
            
            # NEW: Teaching moments (placeholder - requires manual addition)
            "teachingMoments": [],
            
            # NEW: Expression cues (placeholder)
            "expressionCues": [],
            
            # NEW: Tone (placeholder)
            "tone": {
                "voice_character": "enthusiastic_guide",
                "emotional_temperature": "high_energy",
                "language_patterns": {
                    "openings": ["Let's learn together!"],
                    "transitions": ["Now let's see..."],
                    "encouragements": ["You're doing great!"],
                    "closings": ["See you next time!"]
                },
                "metaphor_style": "simple_everyday_examples",
                "question_approach": "open_ended_curious",
                "validation_style": "positive_encouraging"
            }
        }
        
        v2_data["ageVariants"][age_key] = v2_variant
        print(f"   ✅ Migrated age variant: {age_key}")
    
    # Generate output filename
    output_file = input_file.replace('.json', '-dna.json')
    if output_file == input_file:
        output_file = input_file.replace('.json', '-v2-dna.json')
    
    # Write V2 file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(v2_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Migration complete: {output_file}")
    print(f"\n⚠️  MANUAL ENRICHMENT REQUIRED:")
    print(f"   1. Update universal_concept (line ~17)")
    print(f"   2. Update core_principle (line ~25)")
    print(f"   3. Update learning_essence (line ~33)")
    print(f"   4. Add ES/FR translations for above")
    print(f"   5. Add core_metaphor for each age variant")
    print(f"   6. Add teachingMoments (2-4 per age)")
    print(f"   7. Add expressionCues linked to moments")
    print(f"   8. Enrich tone.language_patterns")
    print(f"   9. Translate EN content to ES/FR")
    print(f"   10. Validate with: node validate-lesson.js {output_file}\n")
    
    return output_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python migrate-v1-to-v2.py <lesson-file.json>")
        print("\nExample:")
        print("  python migrate-v1-to-v2.py water-cycle.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"❌ Error: File not found: {input_file}")
        sys.exit(1)
    
    try:
        output_file = migrate_lesson_v1_to_v2(input_file)
        print(f"\n🎉 Success! Migrated lesson saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

