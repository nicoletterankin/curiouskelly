#!/usr/bin/env python3
"""
ElevenLabs Training Data Generator for Kelly25 Voice
Generates hours of training data using ElevenLabs API for Piper TTS training
"""

import os
import json
import time
import requests
import librosa
import soundfile as sf
from pathlib import Path
import pandas as pd
from typing import List, Dict, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ElevenLabsTrainingGenerator:
    """Generate training data using ElevenLabs API for Kelly25 voice"""
    
    def __init__(self, api_key: str, voice_id: str):
        self.api_key = api_key
        self.voice_id = voice_id
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
        # Create output directories
        self.output_dir = Path("kelly25_training_data")
        self.audio_dir = self.output_dir / "wavs"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ElevenLabs generator for voice: {voice_id}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def generate_training_texts(self) -> List[Dict[str, str]]:
        """Generate diverse training text samples for Kelly25 voice"""
        
        training_texts = [
            # Basic greetings and introductions
            {"category": "greeting", "text": "Hello! I'm Kelly, your friendly learning companion."},
            {"category": "greeting", "text": "Hi there! Welcome to our learning adventure together."},
            {"category": "greeting", "text": "Good morning! I'm excited to help you learn today."},
            {"category": "greeting", "text": "Hey! I'm Kelly, and I love helping students discover new things."},
            
            # Educational content
            {"category": "education", "text": "Let's explore something exciting together today!"},
            {"category": "education", "text": "Learning is a journey, and I'm here to guide you."},
            {"category": "education", "text": "Every question you ask brings us closer to understanding."},
            {"category": "education", "text": "I believe in your potential to learn and grow."},
            
            # Encouragement and motivation
            {"category": "encouragement", "text": "You're doing great! Keep up the excellent work."},
            {"category": "encouragement", "text": "Don't worry if something seems difficult at first."},
            {"category": "encouragement", "text": "Every expert was once a beginner, just like you."},
            {"category": "encouragement", "text": "I'm proud of your progress and determination."},
            
            # Questions and interactions
            {"category": "question", "text": "What would you like to learn about today?"},
            {"category": "question", "text": "How are you feeling about this lesson so far?"},
            {"category": "question", "text": "Do you have any questions about what we've covered?"},
            {"category": "question", "text": "What interests you most about this topic?"},
            
            # Explanations and teaching
            {"category": "explanation", "text": "Let me explain this concept in a simple way."},
            {"category": "explanation", "text": "Think of it like this: imagine you're building with blocks."},
            {"category": "explanation", "text": "The key to understanding is breaking things into smaller parts."},
            {"category": "explanation", "text": "Once you see the pattern, everything becomes clearer."},
            
            # Emotional expressions
            {"category": "excitement", "text": "Wow! That's an amazing discovery you just made!"},
            {"category": "excitement", "text": "I'm so excited to see what you'll learn next!"},
            {"category": "excitement", "text": "This is going to be so much fun to explore together!"},
            {"category": "excitement", "text": "You're absolutely brilliant! I love your thinking!"},
            
            # Calm and soothing
            {"category": "calm", "text": "Take your time, there's no rush in learning."},
            {"category": "calm", "text": "Breathe deeply and let's approach this step by step."},
            {"category": "calm", "text": "It's okay to feel confused sometimes, that's part of learning."},
            {"category": "calm", "text": "I'm here to support you through this journey."},
            
            # Technical and academic
            {"category": "technical", "text": "The fundamental principles behind this concept are fascinating."},
            {"category": "technical", "text": "Let's analyze this problem systematically and methodically."},
            {"category": "technical", "text": "The data suggests we should consider multiple perspectives."},
            {"category": "technical", "text": "This approach will help us understand the underlying mechanisms."},
            
            # Storytelling and examples
            {"category": "story", "text": "Once upon a time, there was a curious student just like you."},
            {"category": "story", "text": "Let me tell you a story that illustrates this concept perfectly."},
            {"category": "story", "text": "Imagine you're walking through a magical forest of knowledge."},
            {"category": "story", "text": "Picture this scenario: you're building a bridge across a river."},
            
            # Problem-solving
            {"category": "problem_solving", "text": "Let's work through this challenge together, step by step."},
            {"category": "problem_solving", "text": "What if we tried a different approach to solve this?"},
            {"category": "problem_solving", "text": "Breaking this down into smaller parts will help us succeed."},
            {"category": "problem_solving", "text": "Every problem has a solution, we just need to find it."},
            
            # Reflection and summary
            {"category": "reflection", "text": "Let's take a moment to reflect on what we've learned."},
            {"category": "reflection", "text": "What was the most interesting part of our lesson today?"},
            {"category": "reflection", "text": "I hope you feel more confident about this topic now."},
            {"category": "reflection", "text": "Remember, learning is a continuous process of growth."},
            
            # Future and goals
            {"category": "future", "text": "I can't wait to see what you'll accomplish next!"},
            {"category": "future", "text": "Your curiosity will take you to amazing places."},
            {"category": "future", "text": "I believe you have the potential to achieve great things."},
            {"category": "future", "text": "Let's set some goals for our next learning adventure."},
            
            # Conversational and casual
            {"category": "casual", "text": "You know what? I think you're really getting the hang of this!"},
            {"category": "casual", "text": "That's a great question! I'm glad you asked."},
            {"category": "casual", "text": "Honestly, I learn something new from you every day too."},
            {"category": "casual", "text": "You make teaching so much fun and rewarding for me."},
            
            # Challenges and growth
            {"category": "challenge", "text": "This might seem challenging, but I know you can handle it."},
            {"category": "challenge", "text": "Growth happens when we step outside our comfort zone."},
            {"category": "challenge", "text": "Every challenge is an opportunity to become stronger."},
            {"category": "challenge", "text": "I believe in your ability to overcome any obstacle."},
            
            # Celebrations and achievements
            {"category": "celebration", "text": "Congratulations! You've mastered this concept beautifully!"},
            {"category": "celebration", "text": "I'm so proud of how hard you've worked on this!"},
            {"category": "celebration", "text": "Look how far you've come! Your progress is incredible!"},
            {"category": "celebration", "text": "You should be proud of yourself for this achievement!"},
        ]
        
        # Generate variations and expand the dataset
        expanded_texts = []
        
        # Add variations for each base text
        for base_text in training_texts:
            expanded_texts.append(base_text)
            
            # Add variations with different emotional tones
            variations = [
                f"{base_text['text']} I'm really excited about this!",
                f"{base_text['text']} Take your time with this.",
                f"{base_text['text']} This is going to be so much fun!",
                f"{base_text['text']} I'm here to help you every step of the way.",
                f"{base_text['text']} You're doing an amazing job!",
            ]
            
            for variation in variations:
                expanded_texts.append({
                    "category": base_text["category"],
                    "text": variation
                })
        
        # Add more diverse content with longer, more detailed texts
        additional_texts = [
            # Numbers and counting - Extended
            {"category": "numbers", "text": "Let's count from one to ten together. One, two, three, four, five, six, seven, eight, nine, ten. Counting helps us understand quantity and order."},
            {"category": "numbers", "text": "The number five comes after four and before six. In mathematics, we use numbers to solve problems and understand patterns in the world around us."},
            {"category": "numbers", "text": "Mathematics is like a beautiful puzzle waiting to be solved. Each problem teaches us something new about logic, patterns, and the way our universe works."},
            {"category": "numbers", "text": "Addition and subtraction are fundamental operations that help us understand how quantities change. When we add, we combine amounts, and when we subtract, we take away."},
            {"category": "numbers", "text": "Multiplication is repeated addition, and division is the opposite of multiplication. These operations help us work with larger numbers efficiently."},
            
            # Colors and descriptions - Extended
            {"category": "colors", "text": "The sky is a beautiful shade of blue today, with fluffy white clouds drifting slowly across the horizon. Colors help us describe and appreciate the beauty around us."},
            {"category": "colors", "text": "Red is the color of passion and energy, like a vibrant sunset or a blooming rose. It can represent love, excitement, or even warning signs."},
            {"category": "colors", "text": "Green represents growth, nature, and new beginnings. It's the color of leaves, grass, and the promise of spring after a long winter."},
            {"category": "colors", "text": "Yellow brings sunshine and happiness to our world. It's the color of daffodils, lemons, and the warm glow of morning light."},
            {"category": "colors", "text": "Purple combines the calm of blue with the energy of red, creating a color of mystery, creativity, and imagination."},
            
            # Time and seasons - Extended
            {"category": "time", "text": "Spring is a time of renewal and fresh starts. Flowers bloom, birds return from their winter homes, and the world awakens from its slumber."},
            {"category": "time", "text": "Summer brings warmth and endless possibilities. Long days filled with sunshine, outdoor adventures, and time to explore and play."},
            {"category": "time", "text": "Autumn is a season of change and beautiful colors. Leaves turn golden and red, creating a spectacular display before they fall to the ground."},
            {"category": "time", "text": "Winter is a time for reflection and cozy moments. Snow blankets the earth, and we gather indoors to share warmth and stories."},
            {"category": "time", "text": "Each day has twenty-four hours, and we use clocks to measure time. Time helps us organize our activities and understand when things happen."},
            
            # Science and nature - Extended
            {"category": "science", "text": "The water cycle is a fascinating natural process. Water evaporates from oceans, forms clouds, and falls as rain, nourishing the earth and all living things."},
            {"category": "science", "text": "Plants use sunlight to create their own food through photosynthesis. They take in carbon dioxide and release oxygen, helping us breathe."},
            {"category": "science", "text": "The human body is an incredible machine with bones, muscles, organs, and systems that work together to keep us alive and healthy."},
            {"category": "science", "text": "Stars are born from clouds of gas and dust in space. They shine for billions of years, providing light and energy to their planets."},
            {"category": "science", "text": "Gravity is the invisible force that keeps our feet on the ground and holds planets in their orbits around the sun."},
            {"category": "science", "text": "Sound travels through air as waves, vibrating our eardrums so we can hear music, voices, and all the sounds around us."},
            
            # History and culture - Extended
            {"category": "history", "text": "Learning about the past helps us understand the present and make better decisions for the future. History teaches us about human achievements and mistakes."},
            {"category": "history", "text": "Every culture has unique traditions and stories that have been passed down through generations, creating rich tapestries of human experience."},
            {"category": "history", "text": "Great leaders inspire others to achieve their dreams and work together for common goals. They show us what's possible when we believe in ourselves."},
            {"category": "history", "text": "Ancient civilizations built amazing structures like pyramids and temples, showing us what humans can accomplish when they work together."},
            {"category": "history", "text": "Explorers throughout history have discovered new lands and cultures, expanding our understanding of the world and its diverse peoples."},
            
            # Arts and creativity - Extended
            {"category": "arts", "text": "Art allows us to express ourselves in unique ways, whether through painting, drawing, sculpture, or other creative forms."},
            {"category": "arts", "text": "Music has the power to touch our hearts and souls, bringing people together and expressing emotions that words cannot capture."},
            {"category": "arts", "text": "Creativity is like a muscle that grows stronger with use. The more we practice being creative, the more innovative we become."},
            {"category": "arts", "text": "Dance combines movement, rhythm, and expression to tell stories and convey emotions through the beauty of human motion."},
            {"category": "arts", "text": "Literature opens doors to new worlds and perspectives, allowing us to experience lives and adventures beyond our own."},
            
            # Technology and innovation - Extended
            {"category": "technology", "text": "Technology helps us solve problems and connect with others around the world. It makes our lives easier and opens up new possibilities."},
            {"category": "technology", "text": "Innovation comes from thinking outside the box and finding new ways to solve old problems or create something entirely new."},
            {"category": "technology", "text": "The future is full of exciting possibilities as we develop new technologies that will change how we live, work, and communicate."},
            {"category": "technology", "text": "Computers and the internet have revolutionized how we learn, work, and connect with people from all corners of the globe."},
            {"category": "technology", "text": "Robots and artificial intelligence are becoming more advanced, helping us with tasks and opening up new frontiers in science and medicine."},
            
            # Health and wellness - Extended
            {"category": "health", "text": "Taking care of our bodies helps us think more clearly and have more energy to pursue our goals and dreams."},
            {"category": "health", "text": "Exercise is not just good for our bodies, but our minds too. It releases endorphins that make us feel happy and energized."},
            {"category": "health", "text": "A healthy mind and body work together beautifully. When we feel good physically, we're more likely to feel good mentally too."},
            {"category": "health", "text": "Eating nutritious foods gives our bodies the fuel they need to function properly and helps us grow strong and healthy."},
            {"category": "health", "text": "Getting enough sleep is essential for our health. It helps our bodies repair themselves and our minds process the day's experiences."},
            
            # Friendship and relationships - Extended
            {"category": "relationships", "text": "Good friends support each other through thick and thin, celebrating successes and offering comfort during difficult times."},
            {"category": "relationships", "text": "Kindness is a language that everyone understands, regardless of where they come from or what language they speak."},
            {"category": "relationships", "text": "Listening is one of the greatest gifts we can give to others. It shows we care and helps us understand different perspectives."},
            {"category": "relationships", "text": "Family provides a foundation of love and support that helps us grow and face life's challenges with confidence."},
            {"category": "relationships", "text": "Building trust takes time and consistency. When we keep our promises and treat others with respect, we create lasting bonds."},
            
            # Dreams and aspirations - Extended
            {"category": "dreams", "text": "Dreams are the seeds of tomorrow's achievements. They give us something to work toward and help us imagine what's possible."},
            {"category": "dreams", "text": "Believe in yourself and your ability to make a difference. Every great accomplishment started with someone believing it was possible."},
            {"category": "dreams", "text": "Every great journey begins with a single step. Don't be afraid to start small, because small steps lead to big changes."},
            {"category": "dreams", "text": "Setting goals helps us turn our dreams into reality. When we know where we want to go, we can create a plan to get there."},
            {"category": "dreams", "text": "Persistence is key to achieving our dreams. When we face obstacles, we must keep going and find new ways to overcome them."},
            
            # Extended educational content
            {"category": "education", "text": "Learning is a lifelong journey that never ends. Every day brings new opportunities to discover something fascinating about our world."},
            {"category": "education", "text": "Questions are the keys that unlock knowledge. When we ask questions, we open doors to understanding and discovery."},
            {"category": "education", "text": "Reading opens up entire worlds of knowledge and imagination. Books are like windows that let us see into different times and places."},
            {"category": "education", "text": "Writing helps us organize our thoughts and share our ideas with others. It's a powerful tool for communication and self-expression."},
            {"category": "education", "text": "Critical thinking helps us analyze information and make good decisions. It's an essential skill for navigating our complex world."},
            
            # Extended emotional content
            {"category": "emotions", "text": "Happiness is a wonderful feeling that spreads to others around us. When we're happy, we can't help but share that joy with the world."},
            {"category": "emotions", "text": "Sadness is a natural emotion that helps us process difficult experiences. It's okay to feel sad sometimes, and it's important to talk about our feelings."},
            {"category": "emotions", "text": "Anger can be a powerful emotion, but it's important to express it in healthy ways that don't hurt ourselves or others."},
            {"category": "emotions", "text": "Fear can protect us from danger, but sometimes it can also hold us back from trying new things and growing as people."},
            {"category": "emotions", "text": "Love is one of the most powerful emotions we can experience. It connects us to others and makes life meaningful and beautiful."},
            
            # Extended problem-solving content
            {"category": "problem_solving", "text": "When we face a problem, the first step is to understand what we're dealing with. Then we can brainstorm solutions and try different approaches."},
            {"category": "problem_solving", "text": "Sometimes the best solution isn't the first one we think of. It's important to consider multiple options and choose the one that works best."},
            {"category": "problem_solving", "text": "Working together with others can help us solve problems more effectively. Different perspectives bring different ideas and solutions."},
            {"category": "problem_solving", "text": "Mistakes are not failures, but opportunities to learn and improve. Every mistake teaches us something valuable."},
            {"category": "problem_solving", "text": "Persistence is crucial when solving difficult problems. Sometimes we need to try many different approaches before finding the right solution."},
        ]
        
        expanded_texts.extend(additional_texts)
        
        logger.info(f"Generated {len(expanded_texts)} training text samples")
        return expanded_texts
    
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
    
    def generate_training_dataset(self, target_duration_hours: float = 2.0) -> bool:
        """Generate complete training dataset"""
        
        logger.info(f"Starting generation of {target_duration_hours} hours of training data")
        
        # Generate training texts
        training_texts = self.generate_training_texts()
        
        # Calculate target number of samples
        target_duration_seconds = target_duration_hours * 3600
        avg_duration_per_sample = 12.0  # Average 12 seconds per sample (longer texts)
        target_samples = int(target_duration_seconds / avg_duration_per_sample)
        
        logger.info(f"Target: {target_samples} samples (~{target_duration_hours} hours)")
        
        # Generate audio samples
        successful_samples = 0
        failed_samples = 0
        
        for i, text_data in enumerate(training_texts):
            if successful_samples >= target_samples:
                break
            
            # Create unique filename
            sample_id = f"kelly25_{i+1:04d}"
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
        
        # Create metadata.csv
        self.create_metadata_csv(training_texts[:successful_samples])
        
        logger.info(f"Generation complete: {successful_samples} successful, {failed_samples} failed")
        return successful_samples > 0
    
    def create_metadata_csv(self, text_data: List[Dict[str, str]]):
        """Create metadata.csv file for Piper training"""
        
        metadata_path = self.output_dir / "metadata.csv"
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write("# Kelly25 Training Dataset\n")
            f.write("# Format: id|normalized_text|raw_text\n")
            
            for i, data in enumerate(text_data):
                sample_id = f"kelly25_{i+1:04d}"
                normalized_text = data["text"].upper()
                raw_text = data["text"]
                
                f.write(f"{sample_id}|{normalized_text}|{raw_text}\n")
        
        logger.info(f"Created metadata.csv with {len(text_data)} samples")
    
    def estimate_credits_usage(self, target_duration_hours: float = 2.0) -> int:
        """Estimate ElevenLabs credits needed"""
        
        # Generate training texts to get character count
        training_texts = self.generate_training_texts()
        
        total_characters = sum(len(text["text"]) for text in training_texts)
        avg_chars_per_sample = total_characters / len(training_texts)
        
        target_duration_seconds = target_duration_hours * 3600
        avg_duration_per_sample = 12.0
        target_samples = int(target_duration_seconds / avg_duration_per_sample)
        
        estimated_credits = int(target_samples * avg_chars_per_sample)
        
        logger.info(f"Estimated credits needed: {estimated_credits:,}")
        logger.info(f"Your available credits: 999,000")
        logger.info(f"Remaining after generation: {999000 - estimated_credits:,}")
        
        return estimated_credits

def main():
    """Main function to generate training data"""
    
    # Configuration
    API_KEY = "sk_17b7a1d5b54e992c687a165646ddf84dd3997cd748127568"
    VOICE_ID = "wAdymQH5YucAkXwmrdL0"  # Kelly25 voice
    TARGET_HOURS = 3.0  # Generate 3 hours of training data (60+ minutes minimum)
    
    # Initialize generator
    generator = ElevenLabsTrainingGenerator(API_KEY, VOICE_ID)
    
    # Estimate credits usage
    estimated_credits = generator.estimate_credits_usage(TARGET_HOURS)
    
    if estimated_credits > 999000:
        logger.warning(f"Estimated credits ({estimated_credits:,}) exceed available (999,000)")
        logger.info("Reducing target duration to fit available credits")
        TARGET_HOURS = (999000 / estimated_credits) * TARGET_HOURS
        logger.info(f"Adjusted target: {TARGET_HOURS:.1f} hours")
    
    # Generate training dataset
    success = generator.generate_training_dataset(TARGET_HOURS)
    
    if success:
        logger.info("✅ Training data generation completed successfully!")
        logger.info(f"📁 Output directory: {generator.output_dir}")
        logger.info("📝 Next steps:")
        logger.info("1. Review generated audio files")
        logger.info("2. Prepare for Piper training")
        logger.info("3. Run Piper training pipeline")
    else:
        logger.error("❌ Training data generation failed!")

if __name__ == "__main__":
    main()
