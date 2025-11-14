#!/usr/bin/env python3
"""
Update Metadata for Kelly25 Training Data
Add new samples to metadata.csv
"""

import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_metadata():
    """Update metadata.csv with new samples"""
    
    print("📝 Updating Kelly25 Training Data Metadata")
    print("=" * 50)
    
    # Paths
    data_dir = Path("kelly25_training_data")
    metadata_file = data_dir / "metadata.csv"
    wavs_dir = data_dir / "wavs"
    
    # Load existing metadata
    print("📊 Loading existing metadata...")
    try:
        existing_metadata = pd.read_csv(metadata_file, sep='|', comment='#', names=['id', 'normalized_text', 'raw_text'])
        print(f"✅ Loaded {len(existing_metadata)} existing entries")
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return False
    
    # Get all WAV files
    audio_files = list(wavs_dir.glob("*.wav"))
    print(f"✅ Found {len(audio_files)} WAV files")
    
    # Check if we need to add new entries
    existing_ids = set(existing_metadata['id'].tolist())
    new_files = []
    
    for audio_file in audio_files:
        file_id = audio_file.stem  # Remove .wav extension
        if file_id not in existing_ids:
            new_files.append(file_id)
    
    print(f"📋 Found {len(new_files)} new files to add to metadata")
    
    if len(new_files) == 0:
        print("✅ Metadata is up to date!")
        return True
    
    # Generate text content for new files
    print("📝 Generating text content for new files...")
    
    # Extended text samples (same as in generate_additional_training_data.py)
    extended_texts = [
        "Welcome to our interactive learning session! Today we're going to explore fascinating concepts together. I'm here to guide you through each step, answer your questions, and help you understand complex ideas in simple ways. Learning should be enjoyable and engaging, so don't hesitate to ask questions or share your thoughts.",
        "Let's dive into this topic with curiosity and enthusiasm. Every question you ask brings us closer to understanding. Remember, there are no silly questions - only opportunities to learn something new. I believe in your potential to grasp these concepts, and I'm excited to see your progress as we work through this material together.",
        "Today's lesson is designed to build your confidence and knowledge step by step. We'll start with the basics and gradually work our way up to more advanced concepts. Take your time with each section, and don't worry if something seems challenging at first. That's completely normal, and I'm here to help you through any difficulties you might encounter.",
        "When we encounter a challenging problem, the first step is to break it down into smaller, manageable parts. Let's analyze this step by step, considering different approaches and strategies. Sometimes the best solution isn't immediately obvious, but by thinking creatively and systematically, we can find effective ways to solve even the most complex challenges.",
        "Problem-solving is like being a detective - we gather clues, analyze evidence, and piece together the solution. Don't be discouraged if the first approach doesn't work. Every attempt teaches us something valuable and brings us closer to the answer. Let's explore multiple strategies and see which one works best for this particular situation.",
        "Once upon a time, in a magical kingdom of knowledge, there lived a curious student just like you. This student had a special gift - the ability to see connections between different ideas and concepts. As we journey through this story together, you'll discover how seemingly unrelated topics can come together to create beautiful understanding and wisdom.",
        "Let me tell you a story that illustrates this concept perfectly. In a world where learning was an adventure, students discovered that every challenge was actually an opportunity in disguise. They learned that persistence, creativity, and collaboration were the keys to unlocking any mystery or solving any problem that came their way.",
        "The natural world is full of incredible mysteries waiting to be discovered. From the tiniest particles to the vastness of space, science helps us understand how everything works together. Today we'll explore some of these fascinating phenomena and see how scientific thinking can help us make sense of the world around us.",
        "Science is not just about memorizing facts - it's about asking questions, making observations, and drawing conclusions based on evidence. We'll learn how scientists approach problems, design experiments, and use critical thinking to understand complex phenomena. This process of discovery is what makes science so exciting and rewarding.",
        "Mathematics is the language of patterns and relationships. When we learn to speak this language, we can understand the hidden structures that govern our world. Numbers, shapes, and equations are not just abstract concepts - they're tools that help us solve real-world problems and make sense of complex situations.",
        "Let's explore mathematical thinking together. We'll discover how logical reasoning, pattern recognition, and creative problem-solving come together to help us understand mathematical concepts. Remember, every mathematician was once a beginner, and every expert started with curiosity and determination.",
        "History is not just about dates and events - it's about understanding how people lived, thought, and shaped the world we know today. By studying the past, we can learn valuable lessons about human nature, society, and the consequences of our actions. This knowledge helps us make better decisions for the future.",
        "Every culture has unique traditions, values, and ways of understanding the world. By learning about different cultures and historical periods, we develop empathy, broaden our perspectives, and appreciate the rich diversity of human experience. This understanding helps us become more thoughtful and compassionate members of our global community.",
        "Art is a powerful form of human expression that transcends language and culture. Through painting, music, dance, literature, and other creative forms, artists communicate emotions, ideas, and experiences that words alone cannot capture. Art has the power to inspire, challenge, and transform both the creator and the audience.",
        "Creativity is not limited to traditional artistic pursuits - it's a way of thinking that can be applied to any field or challenge. When we approach problems with creativity, we open ourselves to new possibilities and innovative solutions. Creative thinking involves curiosity, flexibility, and the courage to try new approaches.",
        "Technology is constantly evolving, creating new possibilities and challenges for our society. From artificial intelligence to renewable energy, technological advances are reshaping how we live, work, and interact with each other. Understanding these changes helps us navigate the future with wisdom and responsibility.",
        "Innovation comes from combining existing knowledge in new ways and thinking outside conventional boundaries. The most successful innovations often arise from identifying unmet needs and developing creative solutions. This process requires both technical knowledge and human insight into what people truly need and want.",
        "Taking care of our physical and mental health is essential for living a fulfilling life. This involves making conscious choices about nutrition, exercise, sleep, and stress management. When we prioritize our well-being, we have more energy, focus, and resilience to pursue our goals and help others.",
        "Mental health is just as important as physical health. It's okay to experience a range of emotions, and it's important to develop healthy coping strategies for managing stress and challenges. Seeking support when needed is a sign of strength, not weakness, and taking care of our mental well-being benefits everyone around us.",
        "Healthy relationships are built on trust, respect, and effective communication. When we listen actively, express ourselves clearly, and show empathy toward others, we create strong connections that enrich our lives. Good relationships require effort and understanding, but they provide support, joy, and meaning that make life worthwhile.",
        "Communication is more than just exchanging words - it's about understanding and being understood. This involves not only speaking clearly but also listening with an open mind and heart. When we communicate effectively, we can resolve conflicts, build trust, and create deeper connections with the people in our lives.",
        "Personal growth is a lifelong journey of self-discovery and improvement. It involves setting goals, learning from experiences, and developing new skills and perspectives. Growth often happens outside our comfort zones, so it's important to embrace challenges and view setbacks as opportunities to learn and become stronger.",
        "Self-reflection is a powerful tool for personal development. By regularly examining our thoughts, feelings, and actions, we can identify patterns, recognize areas for improvement, and celebrate our progress. This process of self-awareness helps us make more intentional choices and live more authentically.",
        "Our planet is a complex system where all living things are interconnected. Understanding environmental science helps us appreciate the delicate balance that sustains life on Earth. By learning about ecosystems, climate, and conservation, we can make informed decisions that protect our environment for future generations.",
        "Environmental stewardship is everyone's responsibility. Small actions, when multiplied by millions of people, can create significant positive change. By reducing waste, conserving resources, and supporting sustainable practices, we contribute to a healthier planet and a better future for all living things.",
        "In our interconnected world, being a global citizen means understanding how our actions affect people and communities around the world. This involves learning about different cultures, global issues, and our shared humanity. By developing global awareness, we can contribute to solving problems that transcend national boundaries.",
        "Cultural diversity is one of humanity's greatest strengths. When we learn about different cultures, traditions, and perspectives, we expand our understanding of what it means to be human. This knowledge helps us build bridges between communities and work together to address global challenges.",
        "Critical thinking is the ability to analyze information objectively, evaluate evidence, and make reasoned judgments. This skill is essential for navigating our complex world, where we're constantly bombarded with information from various sources. By developing critical thinking skills, we can distinguish between fact and opinion, identify bias, and make informed decisions.",
        "Analysis involves breaking down complex information into its component parts and examining how they relate to each other. This process helps us understand underlying patterns, identify cause-and-effect relationships, and draw meaningful conclusions. Strong analytical skills are valuable in every field and aspect of life.",
        "Ethics involves thinking about what is right and wrong, and how we should treat others and make decisions. These questions don't always have clear answers, but by considering different perspectives and values, we can develop our own moral compass. Ethical thinking helps us navigate complex situations and make choices that align with our values.",
        "Values are the principles that guide our behavior and decision-making. They shape how we treat others, what we prioritize, and how we define success. By reflecting on our values and understanding how they influence our choices, we can live more intentionally and authentically."
    ]
    
    # Create new metadata entries
    new_entries = []
    for i, file_id in enumerate(new_files):
        # Use extended text, cycling through if we have more files than texts
        text_index = i % len(extended_texts)
        raw_text = extended_texts[text_index]
        normalized_text = raw_text.upper()
        
        new_entries.append({
            'id': file_id,
            'normalized_text': normalized_text,
            'raw_text': raw_text
        })
    
    print(f"✅ Generated {len(new_entries)} new metadata entries")
    
    # Combine existing and new metadata
    new_metadata_df = pd.DataFrame(new_entries)
    combined_metadata = pd.concat([existing_metadata, new_metadata_df], ignore_index=True)
    
    # Sort by ID to maintain order
    combined_metadata = combined_metadata.sort_values('id')
    
    print(f"📊 Total metadata entries: {len(combined_metadata)}")
    
    # Save updated metadata
    print("💾 Saving updated metadata...")
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write("# Kelly25 Training Dataset\n")
        f.write("# Format: id|normalized_text|raw_text\n")
        
        for _, row in combined_metadata.iterrows():
            f.write(f"{row['id']}|{row['normalized_text']}|{row['raw_text']}\n")
    
    print(f"✅ Updated metadata saved to: {metadata_file}")
    print(f"📈 Added {len(new_entries)} new entries")
    print(f"📊 Total entries: {len(combined_metadata)}")
    
    return True

if __name__ == "__main__":
    update_metadata()





































