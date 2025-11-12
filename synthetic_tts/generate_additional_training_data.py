#!/usr/bin/env python3
"""
Generate Additional Kelly25 Training Data
Create more training samples to reach 60+ minutes total
"""

import os
import json
import time
import requests
import librosa
import soundfile as sf
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdditionalTrainingGenerator:
    """Generate additional training data to reach 60+ minutes"""
    
    def __init__(self, api_key: str, voice_id: str):
        self.api_key = api_key
        self.voice_id = voice_id
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
        # Use existing output directory
        self.output_dir = Path("kelly25_training_data")
        self.audio_dir = self.output_dir / "wavs"
        
        logger.info(f"Initialized additional generator for voice: {voice_id}")
    
    def generate_extended_texts(self) -> list:
        """Generate additional extended training texts"""
        
        extended_texts = [
            # Educational scenarios - Extended
            {"category": "education", "text": "Welcome to our interactive learning session! Today we're going to explore fascinating concepts together. I'm here to guide you through each step, answer your questions, and help you understand complex ideas in simple ways. Learning should be enjoyable and engaging, so don't hesitate to ask questions or share your thoughts."},
            
            {"category": "education", "text": "Let's dive into this topic with curiosity and enthusiasm. Every question you ask brings us closer to understanding. Remember, there are no silly questions - only opportunities to learn something new. I believe in your potential to grasp these concepts, and I'm excited to see your progress as we work through this material together."},
            
            {"category": "education", "text": "Today's lesson is designed to build your confidence and knowledge step by step. We'll start with the basics and gradually work our way up to more advanced concepts. Take your time with each section, and don't worry if something seems challenging at first. That's completely normal, and I'm here to help you through any difficulties you might encounter."},
            
            # Problem-solving scenarios - Extended
            {"category": "problem_solving", "text": "When we encounter a challenging problem, the first step is to break it down into smaller, manageable parts. Let's analyze this step by step, considering different approaches and strategies. Sometimes the best solution isn't immediately obvious, but by thinking creatively and systematically, we can find effective ways to solve even the most complex challenges."},
            
            {"category": "problem_solving", "text": "Problem-solving is like being a detective - we gather clues, analyze evidence, and piece together the solution. Don't be discouraged if the first approach doesn't work. Every attempt teaches us something valuable and brings us closer to the answer. Let's explore multiple strategies and see which one works best for this particular situation."},
            
            # Storytelling and narratives - Extended
            {"category": "story", "text": "Once upon a time, in a magical kingdom of knowledge, there lived a curious student just like you. This student had a special gift - the ability to see connections between different ideas and concepts. As we journey through this story together, you'll discover how seemingly unrelated topics can come together to create beautiful understanding and wisdom."},
            
            {"category": "story", "text": "Let me tell you a story that illustrates this concept perfectly. In a world where learning was an adventure, students discovered that every challenge was actually an opportunity in disguise. They learned that persistence, creativity, and collaboration were the keys to unlocking any mystery or solving any problem that came their way."},
            
            # Science and discovery - Extended
            {"category": "science", "text": "The natural world is full of incredible mysteries waiting to be discovered. From the tiniest particles to the vastness of space, science helps us understand how everything works together. Today we'll explore some of these fascinating phenomena and see how scientific thinking can help us make sense of the world around us."},
            
            {"category": "science", "text": "Science is not just about memorizing facts - it's about asking questions, making observations, and drawing conclusions based on evidence. We'll learn how scientists approach problems, design experiments, and use critical thinking to understand complex phenomena. This process of discovery is what makes science so exciting and rewarding."},
            
            # Mathematics and logic - Extended
            {"category": "math", "text": "Mathematics is the language of patterns and relationships. When we learn to speak this language, we can understand the hidden structures that govern our world. Numbers, shapes, and equations are not just abstract concepts - they're tools that help us solve real-world problems and make sense of complex situations."},
            
            {"category": "math", "text": "Let's explore mathematical thinking together. We'll discover how logical reasoning, pattern recognition, and creative problem-solving come together to help us understand mathematical concepts. Remember, every mathematician was once a beginner, and every expert started with curiosity and determination."},
            
            # History and culture - Extended
            {"category": "history", "text": "History is not just about dates and events - it's about understanding how people lived, thought, and shaped the world we know today. By studying the past, we can learn valuable lessons about human nature, society, and the consequences of our actions. This knowledge helps us make better decisions for the future."},
            
            {"category": "history", "text": "Every culture has unique traditions, values, and ways of understanding the world. By learning about different cultures and historical periods, we develop empathy, broaden our perspectives, and appreciate the rich diversity of human experience. This understanding helps us become more thoughtful and compassionate members of our global community."},
            
            # Arts and creativity - Extended
            {"category": "arts", "text": "Art is a powerful form of human expression that transcends language and culture. Through painting, music, dance, literature, and other creative forms, artists communicate emotions, ideas, and experiences that words alone cannot capture. Art has the power to inspire, challenge, and transform both the creator and the audience."},
            
            {"category": "arts", "text": "Creativity is not limited to traditional artistic pursuits - it's a way of thinking that can be applied to any field or challenge. When we approach problems with creativity, we open ourselves to new possibilities and innovative solutions. Creative thinking involves curiosity, flexibility, and the courage to try new approaches."},
            
            # Technology and innovation - Extended
            {"category": "technology", "text": "Technology is constantly evolving, creating new possibilities and challenges for our society. From artificial intelligence to renewable energy, technological advances are reshaping how we live, work, and interact with each other. Understanding these changes helps us navigate the future with wisdom and responsibility."},
            
            {"category": "technology", "text": "Innovation comes from combining existing knowledge in new ways and thinking outside conventional boundaries. The most successful innovations often arise from identifying unmet needs and developing creative solutions. This process requires both technical knowledge and human insight into what people truly need and want."},
            
            # Health and wellness - Extended
            {"category": "health", "text": "Taking care of our physical and mental health is essential for living a fulfilling life. This involves making conscious choices about nutrition, exercise, sleep, and stress management. When we prioritize our well-being, we have more energy, focus, and resilience to pursue our goals and help others."},
            
            {"category": "health", "text": "Mental health is just as important as physical health. It's okay to experience a range of emotions, and it's important to develop healthy coping strategies for managing stress and challenges. Seeking support when needed is a sign of strength, not weakness, and taking care of our mental well-being benefits everyone around us."},
            
            # Relationships and communication - Extended
            {"category": "relationships", "text": "Healthy relationships are built on trust, respect, and effective communication. When we listen actively, express ourselves clearly, and show empathy toward others, we create strong connections that enrich our lives. Good relationships require effort and understanding, but they provide support, joy, and meaning that make life worthwhile."},
            
            {"category": "relationships", "text": "Communication is more than just exchanging words - it's about understanding and being understood. This involves not only speaking clearly but also listening with an open mind and heart. When we communicate effectively, we can resolve conflicts, build trust, and create deeper connections with the people in our lives."},
            
            # Personal growth and development - Extended
            {"category": "growth", "text": "Personal growth is a lifelong journey of self-discovery and improvement. It involves setting goals, learning from experiences, and developing new skills and perspectives. Growth often happens outside our comfort zones, so it's important to embrace challenges and view setbacks as opportunities to learn and become stronger."},
            
            {"category": "growth", "text": "Self-reflection is a powerful tool for personal development. By regularly examining our thoughts, feelings, and actions, we can identify patterns, recognize areas for improvement, and celebrate our progress. This process of self-awareness helps us make more intentional choices and live more authentically."},
            
            # Environmental awareness - Extended
            {"category": "environment", "text": "Our planet is a complex system where all living things are interconnected. Understanding environmental science helps us appreciate the delicate balance that sustains life on Earth. By learning about ecosystems, climate, and conservation, we can make informed decisions that protect our environment for future generations."},
            
            {"category": "environment", "text": "Environmental stewardship is everyone's responsibility. Small actions, when multiplied by millions of people, can create significant positive change. By reducing waste, conserving resources, and supporting sustainable practices, we contribute to a healthier planet and a better future for all living things."},
            
            # Global citizenship - Extended
            {"category": "global", "text": "In our interconnected world, being a global citizen means understanding how our actions affect people and communities around the world. This involves learning about different cultures, global issues, and our shared humanity. By developing global awareness, we can contribute to solving problems that transcend national boundaries."},
            
            {"category": "global", "text": "Cultural diversity is one of humanity's greatest strengths. When we learn about different cultures, traditions, and perspectives, we expand our understanding of what it means to be human. This knowledge helps us build bridges between communities and work together to address global challenges."},
            
            # Critical thinking and analysis - Extended
            {"category": "critical_thinking", "text": "Critical thinking is the ability to analyze information objectively, evaluate evidence, and make reasoned judgments. This skill is essential for navigating our complex world, where we're constantly bombarded with information from various sources. By developing critical thinking skills, we can distinguish between fact and opinion, identify bias, and make informed decisions."},
            
            {"category": "critical_thinking", "text": "Analysis involves breaking down complex information into its component parts and examining how they relate to each other. This process helps us understand underlying patterns, identify cause-and-effect relationships, and draw meaningful conclusions. Strong analytical skills are valuable in every field and aspect of life."},
            
            # Ethics and values - Extended
            {"category": "ethics", "text": "Ethics involves thinking about what is right and wrong, and how we should treat others and make decisions. These questions don't always have clear answers, but by considering different perspectives and values, we can develop our own moral compass. Ethical thinking helps us navigate complex situations and make choices that align with our values."},
            
            {"category": "ethics", "text": "Values are the principles that guide our behavior and decision-making. They shape how we treat others, what we prioritize, and how we define success. By reflecting on our values and understanding how they influence our choices, we can live more intentionally and authentically."},
        ]
        
        logger.info(f"Generated {len(extended_texts)} extended training text samples")
        return extended_texts
    
    def generate_audio(self, text: str, output_path: str, model_id: str = "eleven_multilingual_v2") -> bool:
        """Generate audio using ElevenLabs API"""
        
        url = f"{self.base_url}/text-to-speech/{self.voice_id}"
        
        data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        try:
            response = requests.post(url, json=data, headers=self.headers)
            response.raise_for_status()
            
            # Save audio file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Generated audio: {output_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to generate audio: {e}")
            return False
    
    def process_audio_file(self, input_path: str, output_path: str) -> bool:
        """Process audio file to match Piper requirements (22kHz, mono)"""
        
        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=22050, mono=True)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Save as WAV
            sf.write(output_path, audio, 22050)
            
            logger.info(f"Processed audio: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process audio: {e}")
            return False
    
    def generate_additional_samples(self, target_additional_minutes: float = 30.0) -> bool:
        """Generate additional training samples to reach target duration"""
        
        logger.info(f"Generating {target_additional_minutes} additional minutes of training data")
        
        # Generate extended texts
        extended_texts = self.generate_extended_texts()
        
        # Calculate target samples (assuming 12 seconds average per sample)
        target_samples = int((target_additional_minutes * 60) / 12.0)
        
        logger.info(f"Target: {target_samples} additional samples")
        
        # Find starting index
        existing_files = list(self.audio_dir.glob("*.wav"))
        start_index = len(existing_files) + 1
        
        # Generate additional audio samples
        successful_samples = 0
        failed_samples = 0
        
        for i, text_data in enumerate(extended_texts):
            if successful_samples >= target_samples:
                break
            
            # Create unique filename
            sample_id = f"kelly25_{start_index + i:04d}"
            mp3_path = self.audio_dir / f"{sample_id}.mp3"
            wav_path = self.audio_dir / f"{sample_id}.wav"
            
            # Generate audio
            if self.generate_audio(text_data["text"], str(mp3_path)):
                # Process to WAV format
                if self.process_audio_file(str(mp3_path), str(wav_path)):
                    successful_samples += 1
                    
                    # Remove MP3 file to save space
                    mp3_path.unlink()
                    
                    # Add delay to respect API limits
                    time.sleep(0.5)
                else:
                    failed_samples += 1
            else:
                failed_samples += 1
            
            # Progress update
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {successful_samples} successful, {failed_samples} failed")
        
        logger.info(f"Additional generation complete: {successful_samples} successful, {failed_samples} failed")
        return successful_samples > 0

def main():
    """Main function to generate additional training data"""
    
    # Configuration
    API_KEY = "sk_17b7a1d5b54e992c687a165646ddf84dd3997cd748127568"
    VOICE_ID = "wAdymQH5YucAkXwmrdL0"  # Kelly25 voice
    ADDITIONAL_MINUTES = 5.0  # Generate 5 additional minutes to reach 60+ total
    
    # Initialize generator
    generator = AdditionalTrainingGenerator(API_KEY, VOICE_ID)
    
    # Generate additional training samples
    success = generator.generate_additional_samples(ADDITIONAL_MINUTES)
    
    if success:
        logger.info("✅ Additional training data generation completed successfully!")
        logger.info("📝 Next steps:")
        logger.info("1. Run validation script to check total duration")
        logger.info("2. Update metadata.csv with new samples")
        logger.info("3. Prepare for Piper training")
    else:
        logger.error("❌ Additional training data generation failed!")

if __name__ == "__main__":
    main()
