"""
Prompt Templates for Content Generation
"""

ARCHETYPE_DESCRIPTIONS = {
    "The Scientist": "analytical, evidence-based, methodical. Uses precise language, cites data, approaches topics with empirical curiosity. Values proof and logical reasoning.",
    "The Explorer": "curious, adventurous, discovery-focused. Frames learning as an expedition, celebrates unknowns, encourages wonder and exploration.",
    "The Storyteller": "narrative-driven, emotional, relatable. Weaves information into stories, uses metaphors, connects facts to human experiences.",
    "The Survivor": "practical, resilient, real-world focused. Emphasizes survival value, practical applications, preparedness. Direct and no-nonsense.",
    "The MacGyver": "creative problem-solver, hands-on, innovative. Loves DIY approaches, asks 'how can we use this?', celebrates ingenuity.",
    "The Empath": "emotionally intelligent, relational, compassionate. Focuses on feelings, relationships, and how topics affect people and communities.",
    "The Rebel": "questioning, challenging assumptions, unconventional. Pushes back on common beliefs, celebrates disruption, encourages critical thinking.",
    "The Architect": "systems thinker, structured, big-picture. Sees connections between parts, loves frameworks, explains how things fit together.",
    "The Mystic": "philosophical, meaning-seeking, contemplative. Explores deeper significance, asks 'why does this matter?', seeks transcendent insights.",
    "The Diplomat": "balanced, multiple perspectives, bridge-builder. Presents various viewpoints fairly, finds common ground, values nuance.",
}

AGE_LANGUAGE_GUIDES = {
    "2-5": {
        "vocabulary": "simple, concrete words. Max 2 syllables preferred. Use sensory words (see, feel, hear).",
        "sentence_length": "5-8 words per sentence. Short, direct.",
        "concepts": "concrete, observable. What you can see, touch, feel. Immediate cause and effect.",
        "tone": "warm, playful, reassuring. Like a friendly teacher. Use exclamations!",
        "examples": "everyday objects (toys, food, animals, family). Things they interact with daily.",
    },
    "6-12": {
        "vocabulary": "expanding vocabulary. Can introduce new words with context clues. Some technical terms OK with explanation.",
        "sentence_length": "10-15 words. Can use compound sentences.",
        "concepts": "cause and effect, basic systems, comparisons. 'This is like that because...'",
        "tone": "encouraging, curious, respectful of their growing knowledge. Treat them as capable learners.",
        "examples": "school, nature, sports, games, books, media they know. Real-world applications.",
    },
    "13-17": {
        "vocabulary": "full vocabulary. Technical terms welcome. Can handle abstraction.",
        "sentence_length": "varied, can be complex. Mirrors adult writing.",
        "concepts": "abstract thinking, hypotheticals, systems thinking, social implications.",
        "tone": "peer-like but not condescending. Respect their intelligence. Can be challenging.",
        "examples": "social media, technology, career, relationships, current events, pop culture.",
    },
    "18-35": {
        "vocabulary": "professional, sophisticated. Industry-specific terms when relevant.",
        "sentence_length": "varied, efficient. Value their time.",
        "concepts": "full complexity. Nuance, trade-offs, real-world applications, career relevance.",
        "tone": "collegial, direct, engaging. Assume competence. Can be witty.",
        "examples": "work, relationships, finance, health, technology, personal development.",
    },
    "36-60": {
        "vocabulary": "mature, precise. Reference broader life experience.",
        "sentence_length": "clear and purposeful. Efficiency valued.",
        "concepts": "application-focused, legacy thinking, mentorship angle, family implications.",
        "tone": "respectful, acknowledging experience. Can reference 'you've seen this evolve.'",
        "examples": "family, career leadership, health, investment, community, passing on knowledge.",
    },
    "61-102": {
        "vocabulary": "dignified, clear. Avoid trendy slang. Classical references OK.",
        "sentence_length": "clear, not rushed. Comfortable pace.",
        "concepts": "wisdom and meaning, legacy, life perspective, historical context.",
        "tone": "warm, dignified, honoring their experience. Never patronizing.",
        "examples": "grandchildren, legacy, historical changes witnessed, wisdom to share, health and vitality.",
    },
}

def get_atom_generation_prompt(lesson: dict, archetype: str, phase: str) -> str:
    """Generate prompt for creating a lesson atom."""
    
    arch_desc = ARCHETYPE_DESCRIPTIONS.get(archetype, "balanced, curious, engaging")
    
    phase_instructions = {
        "Hook": "Create an attention-grabbing opening that draws the learner in. Make them curious to learn more. 2-3 sentences max.",
        "Fact1": "Present the first fascinating fact about this topic. Make it surprising or counter-intuitive if possible. Include a clear explanation.",
        "Fact2": "Build on Fact1 with a second insight that adds depth. Show a different angle or application of the concept.",
        "Fact3": "Deliver a 'wow' moment - the most surprising or delightful fact. This should be memorable and shareable.",
        "Wisdom": "Close with a reflective insight that connects this knowledge to the learner's life. What does this mean for them? How can they apply it?",
    }
    
    return f"""Generate interactive lesson content for Kelly, an AI teacher avatar.

TOPIC: {lesson.get('topic', 'Unknown')}
UNIVERSAL TRUTH: {lesson.get('universal_truth', '')}
LEARNING OBJECTIVES: {lesson.get('learning_objectives', [])}

ARCHETYPE: {archetype}
Archetype personality: {arch_desc}

PHASE: {phase}
Phase goal: {phase_instructions.get(phase, 'Engage and educate.')}

Generate a JSON object with this EXACT structure:
{{
    "script": "What Kelly says to the learner. 2-4 sentences. Written in {archetype} voice.",
    "options": [
        "First response option the learner can choose",
        "Second response option the learner can choose", 
        "Third response option the learner can choose"
    ],
    "responses": {{
        "Option A": "Kelly's response if learner picks option 1",
        "Option B": "Kelly's response if learner picks option 2",
        "Option C": "Kelly's response if learner picks option 3"
    }}
}}

RULES:
1. Script must be in the {archetype} voice/personality
2. Options should represent different learner mindsets (curious, skeptical, playful)
3. Responses should acknowledge the choice and add value
4. Keep it conversational and engaging
5. ONLY output valid JSON, no other text"""


def get_age_variant_prompt(lesson: dict, age_range: str, tone: str) -> str:
    """Generate prompt for creating an age-specific shard."""
    
    age_guide = AGE_LANGUAGE_GUIDES.get(age_range, AGE_LANGUAGE_GUIDES["18-35"])
    
    tone_instructions = {
        "curious": "Express wonder and ask thought-provoking questions. 'I wonder...' 'Have you ever noticed...'",
        "playful": "Use humor, wordplay, and fun energy. Make learning feel like play.",
        "serious": "Be direct and factual. Respect the gravity of knowledge. No fluff.",
        "warm": "Be nurturing and supportive. Create a safe space for learning. Encouraging.",
        "inspiring": "Motivate and uplift. Connect knowledge to possibility and potential.",
    }
    
    return f"""Generate age-appropriate lesson content for Kelly, an AI teacher avatar.

TOPIC: {lesson.get('topic', 'Unknown')}
UNIVERSAL TRUTH: {lesson.get('universal_truth', '')}

TARGET AGE RANGE: {age_range} years old
TONE: {tone}

AGE-APPROPRIATE LANGUAGE GUIDE:
- Vocabulary: {age_guide['vocabulary']}
- Sentence length: {age_guide['sentence_length']}
- Concepts: {age_guide['concepts']}
- Tone: {age_guide['tone']}
- Examples to use: {age_guide['examples']}

TONE INSTRUCTION: {tone_instructions.get(tone, 'Engaging and educational')}

Generate a JSON object with this EXACT structure:
{{
    "script": "What Kelly says. Adapted for {age_range} year olds in a {tone} tone. 2-4 sentences.",
    "options": [
        "Age-appropriate response option 1",
        "Age-appropriate response option 2",
        "Age-appropriate response option 3"
    ],
    "responses": {{
        "Option 1 text here": "Kelly's response to option 1",
        "Option 2 text here": "Kelly's response to option 2",
        "Option 3 text here": "Kelly's response to option 3"
    }}
}}

CRITICAL: Match vocabulary and complexity to {age_range} year olds.
ONLY output valid JSON, no other text."""


def get_translation_prompt(content: dict, target_language: str, language_name: str) -> str:
    """Generate prompt for translating content."""
    
    return f"""Translate this educational content from English to {language_name}.

ORIGINAL CONTENT:
{content}

RULES:
1. Maintain the same JSON structure
2. Translate naturally, not word-for-word
3. Keep the same tone and energy
4. Adapt cultural references if needed
5. Preserve educational intent

OUTPUT: Valid JSON in {language_name}, same structure as input.
ONLY output the translated JSON, no other text."""




