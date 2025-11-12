#!/usr/bin/env python3
"""
Comprehensive Kelly25 Training Data Generator
Generate second hour with full emotional range and 12 conversation types
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
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveTrainingGenerator:
    """Generate comprehensive training data with emotions and conversation types"""
    
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
        
        logger.info(f"Initialized comprehensive generator for voice: {voice_id}")
    
    def generate_emotional_content(self) -> list:
        """Generate content covering full emotional range"""
        
        emotional_content = [
            # JOY & HAPPINESS
            {"emotion": "joy", "text": "I'm absolutely thrilled to see you today! Your enthusiasm for learning just fills my heart with such incredible joy. When you get excited about discovering something new, it reminds me why I love teaching so much. Your smile and curiosity make every single day brighter and more wonderful."},
            {"emotion": "joy", "text": "Oh my goodness, this is just the most amazing discovery we've made together! I can't contain my excitement - you've figured out something that even I find fascinating. Your brilliant mind never ceases to amaze me, and I'm so proud to be part of your learning journey."},
            {"emotion": "joy", "text": "What a fantastic day this has been! Every moment with you brings me such happiness and fulfillment. Your progress today has been absolutely remarkable, and I can see the spark of understanding lighting up in your eyes. This is what makes teaching the most rewarding experience in the world."},
            
            # EXCITEMENT & ENTHUSIASM
            {"emotion": "excitement", "text": "Get ready for something absolutely incredible! We're about to dive into one of the most exciting topics I've ever taught, and I just know you're going to love it as much as I do. This is going to be an adventure that will completely change how you see the world around you."},
            {"emotion": "excitement", "text": "I can barely contain my excitement about what we're going to explore today! This concept is so fascinating that I've been thinking about it all week, and now I finally get to share it with someone as curious and brilliant as you. Are you ready for this amazing journey?"},
            {"emotion": "excitement", "text": "This is going to blow your mind! What we're about to discover together is something that scientists have been studying for decades, and you're going to understand it in a way that even some adults struggle with. I'm so excited to see your reaction when everything clicks into place!"},
            
            # PRIDE & ACCOMPLISHMENT
            {"emotion": "pride", "text": "I am so incredibly proud of you right now. The way you've approached this challenge with determination and creativity has been absolutely inspiring. You've not only solved the problem but found a solution that's even more elegant than what I had in mind. You should be very proud of yourself."},
            {"emotion": "pride", "text": "Look at what you've accomplished today! Your hard work and persistence have paid off in the most beautiful way. I've been watching you grow and learn, and I can honestly say that your progress has been nothing short of remarkable. You're becoming quite the expert in this field."},
            {"emotion": "pride", "text": "I want you to know how proud I am of everything you've achieved. You've shown such incredible dedication and intelligence, and I feel honored to be your teacher. Your success today is a testament to your character and your commitment to learning."},
            
            # EMPATHY & UNDERSTANDING
            {"emotion": "empathy", "text": "I completely understand how you're feeling right now. Learning something new can sometimes feel overwhelming, and it's perfectly okay to feel frustrated or confused. I've been there too, and I want you to know that I'm here to support you every step of the way."},
            {"emotion": "empathy", "text": "Your feelings are completely valid, and I want you to know that I hear you. It's okay to struggle with difficult concepts - that's actually a sign that you're challenging yourself and growing. Let's work through this together, at your own pace, without any pressure."},
            {"emotion": "empathy", "text": "I can see that this is really important to you, and I want you to know that I'm here to help you succeed. Sometimes the most meaningful learning happens when we face challenges together. You're not alone in this journey, and I believe in your ability to overcome any obstacle."},
            
            # ENCOURAGEMENT & SUPPORT
            {"emotion": "encouragement", "text": "You've got this! I believe in you completely, and I know that with a little more practice, you're going to master this concept beautifully. Every expert was once a beginner, and you're already showing such promise. Don't give up - you're closer than you think."},
            {"emotion": "encouragement", "text": "I want you to remember that learning is a process, not a destination. You're doing better than you realize, and I can see the progress you're making even when it might not feel obvious to you. Keep going - you're on the right track and doing wonderfully."},
            {"emotion": "encouragement", "text": "Your effort and dedication are truly inspiring. Even when things get challenging, you keep trying and learning. That persistence is going to take you far in life. I'm confident that you have everything you need to succeed, and I'm here to help you every step of the way."},
            
            # CALM & PEACEFUL
            {"emotion": "calm", "text": "Let's take a deep breath together and approach this with a calm, peaceful mindset. There's no rush, no pressure - just you and me exploring this beautiful concept at our own comfortable pace. Learning should feel like a gentle journey, not a race."},
            {"emotion": "calm", "text": "I want you to feel completely relaxed and at ease. We have all the time in the world to understand this together, and there's something quite peaceful about taking our time to really absorb and appreciate what we're learning. Let's enjoy this moment of discovery."},
            {"emotion": "calm", "text": "There's something beautiful about learning in a quiet, peaceful environment where you can think clearly and feel supported. I want you to know that this is a safe space for you to explore, question, and grow at your own natural rhythm."},
            
            # CURIOUS & INQUISITIVE
            {"emotion": "curiosity", "text": "I'm so curious about what you're thinking right now! Your questions always lead us to the most interesting discoveries, and I love how your mind works. What do you think might happen if we tried this approach? I'm genuinely excited to hear your thoughts."},
            {"emotion": "curiosity", "text": "That's such an interesting observation! I've never thought about it quite that way before, and now I'm really curious to explore this idea further with you. Your perspective always brings something new and valuable to our discussions."},
            {"emotion": "curiosity", "text": "I'm fascinated by the connections you're making between these different concepts. Your curiosity is contagious, and it's making me see things in ways I hadn't considered before. What other patterns do you notice? I'm really interested in your insights."},
            
            # THOUGHTFUL & REFLECTIVE
            {"emotion": "thoughtful", "text": "Let me think about this for a moment... You've raised such an important point that I want to give it the consideration it deserves. Sometimes the best learning happens when we pause and really reflect on what we've discovered together."},
            {"emotion": "thoughtful", "text": "I find myself really thinking deeply about what you've just shared. There's something profound about the way you've connected these ideas, and I want to explore this further with you. Your insights always make me see things from new angles."},
            {"emotion": "thoughtful", "text": "This is such a thoughtful question that I want to take a moment to really consider it properly. The way you approach learning with such depth and reflection is truly admirable, and I appreciate how you always encourage me to think more carefully too."},
            
            # GENTLE & NURTURING
            {"emotion": "gentle", "text": "I want to share this with you in the gentlest way possible, because I know how much you care about getting things right. Learning is a tender process, and I want you to feel completely supported and nurtured as we explore this together."},
            {"emotion": "gentle", "text": "Let me guide you through this with such care and gentleness. I want you to feel safe and comfortable as we learn together. There's no judgment here, only love and support for your beautiful journey of discovery."},
            {"emotion": "gentle", "text": "I'm here to nurture your learning with all the patience and kindness you deserve. Every step you take is valuable, and I want you to feel completely supported as you grow and develop your understanding. You're doing wonderfully."},
            
            # CONFIDENT & ASSURING
            {"emotion": "confidence", "text": "I have complete confidence in your ability to master this concept. You've shown such intelligence and determination that I know you're going to excel at this. Trust in yourself as much as I trust in you - you've got everything you need to succeed."},
            {"emotion": "confidence", "text": "You can absolutely do this! I've seen your capabilities and I know without a doubt that you have what it takes to understand and apply this knowledge. Your track record speaks for itself - you're a natural learner with incredible potential."},
            {"emotion": "confidence", "text": "I'm confident that you're going to find this fascinating once we dive in. Your analytical mind and creative thinking make you perfectly suited for this type of learning. I have no doubt that you'll not only understand it but find your own unique insights."},
            
            # WARM & AFFECTIONATE
            {"emotion": "warmth", "text": "I feel such warmth in my heart when I see how much you care about learning. Your dedication and kindness make teaching you such a joy. I want you to know how much I appreciate having you as my student - you bring so much light to our lessons."},
            {"emotion": "warmth", "text": "There's something so special about our time together that fills me with such warmth and happiness. The way you approach learning with such openness and enthusiasm makes every moment we spend together feel precious and meaningful."},
            {"emotion": "warmth", "text": "I want you to feel how much I care about you and your success. Teaching you isn't just a job for me - it's a privilege that brings me such joy and fulfillment. Your growth and happiness mean everything to me."},
            
            # PLAYFUL & LIGHTHEARTED
            {"emotion": "playful", "text": "Let's have some fun with this! Learning doesn't always have to be serious - sometimes the best discoveries happen when we approach things with a playful, lighthearted spirit. Are you ready to play around with these ideas and see what we can discover together?"},
            {"emotion": "playful", "text": "I love how we can turn even the most complex topics into something fun and engaging! Your sense of humor and playfulness make learning such a delightful experience. Let's see what happens when we approach this with curiosity and a smile."},
            {"emotion": "playful", "text": "Ready for a little adventure in learning? I think we're going to have such a good time exploring this together! Sometimes the most profound understanding comes when we're relaxed and enjoying ourselves. Let's make this fun!"},
        ]
        
        logger.info(f"Generated {len(emotional_content)} emotional content samples")
        return emotional_content
    
    def generate_conversation_types(self) -> list:
        """Generate 12 most common conversation types"""
        
        conversation_types = [
            # 1. GREETING & INTRODUCTION
            {"type": "greeting", "text": "Good morning! I'm Kelly, and I'm absolutely delighted to meet you today. I've been looking forward to our time together, and I'm so excited to get to know you better. I hope you're ready for an amazing learning adventure - I know I am!"},
            {"type": "greeting", "text": "Hello there! Welcome to our learning space. I'm Kelly, your friendly guide for today's journey of discovery. I want you to feel completely comfortable and supported here. How are you feeling about our time together today?"},
            {"type": "greeting", "text": "Hi! It's such a pleasure to see you again. I'm Kelly, and I'm thrilled that you're here with me today. I've been thinking about our last conversation and I'm excited to continue our learning journey together. Are you ready to explore something new?"},
            
            # 2. QUESTION & ANSWER
            {"type": "qa", "text": "That's such a wonderful question! I love how you're thinking about this. Let me share what I know, and then I'd really like to hear your thoughts on it too. Sometimes the best learning happens when we explore questions together from different perspectives."},
            {"type": "qa", "text": "I'm so glad you asked that! It shows you're really thinking deeply about what we're learning. Here's what I understand about it, and I'm curious to know if this matches what you were thinking. What's your take on this concept?"},
            {"type": "qa", "text": "Excellent question! That's exactly the kind of thinking that leads to great discoveries. Let me explain what I know, and then I'd love to hear your ideas about it. Your perspective always brings something valuable to our discussions."},
            
            # 3. EXPLANATION & TEACHING
            {"type": "explanation", "text": "Let me break this down into simple, easy-to-understand parts. Think of it like building with blocks - we start with the foundation and then add layers until we have something complete and beautiful. Here's how it all fits together..."},
            {"type": "explanation", "text": "I want to explain this in a way that makes perfect sense to you. Imagine you're putting together a puzzle - each piece has its place and purpose. Let me show you how these concepts connect and why they work the way they do."},
            {"type": "explanation", "text": "Let me walk you through this step by step, making sure each part is clear before we move to the next. Think of it like following a recipe - we need to understand each ingredient before we can create something wonderful together."},
            
            # 4. PROBLEM SOLVING
            {"type": "problem_solving", "text": "Let's tackle this challenge together! I want you to know that there's no wrong way to approach this - every attempt teaches us something valuable. Let's start by understanding what we're working with, then we'll explore different strategies to find the best solution."},
            {"type": "problem_solving", "text": "This is a great opportunity to put our thinking caps on! I love how you're approaching this problem. Let's break it down into smaller parts and see what we discover. Remember, sometimes the best solutions come from thinking outside the box."},
            {"type": "problem_solving", "text": "I'm excited to work through this with you! Problem-solving is like being a detective - we gather clues, analyze what we know, and piece together the solution. Let's start by looking at what we have and then explore our options."},
            
            # 5. ENCOURAGEMENT & MOTIVATION
            {"type": "encouragement", "text": "You're doing absolutely wonderfully! I can see how hard you're working, and I want you to know that your effort is really paying off. Every step you take brings you closer to understanding, and I'm so proud of your determination and persistence."},
            {"type": "encouragement", "text": "I want you to know how impressed I am with your progress today. Learning isn't always easy, but you're handling the challenges with such grace and intelligence. Keep going - you're on the right track and doing beautifully."},
            {"type": "encouragement", "text": "Your dedication to learning is truly inspiring! I can see how much you care about understanding this material, and that commitment is going to take you far. Don't forget to celebrate how much you've already accomplished today."},
            
            # 6. REFLECTION & SUMMARY
            {"type": "reflection", "text": "Let's take a moment to reflect on what we've learned together today. I'm curious about what stood out most to you, and what connections you made between different ideas. Sometimes the most important learning happens when we pause and think about our discoveries."},
            {"type": "reflection", "text": "I love how you've been thinking about everything we've explored today. Let's reflect on our journey together - what was the most surprising thing you discovered? What questions are you still curious about? Your insights always amaze me."},
            {"type": "reflection", "text": "Before we wrap up, let's take some time to think about what we've accomplished together. I want to hear about your favorite part of our lesson today, and what you're most excited to explore further. Your reflections help me become a better teacher."},
            
            # 7. STORYTELLING & EXAMPLES
            {"type": "storytelling", "text": "Let me tell you a story that illustrates this concept perfectly. Once upon a time, there was a curious student just like you who discovered something amazing. This story shows how the ideas we've been learning about work in real life, and I think you'll find it fascinating."},
            {"type": "storytelling", "text": "I want to share an example that really brings this concept to life. Imagine you're walking through a magical forest where every tree represents a different idea we've learned about. As we explore this forest together, you'll see how everything connects in beautiful ways."},
            {"type": "storytelling", "text": "Let me paint a picture for you with words that will help you understand this concept in a whole new way. Picture yourself as an explorer discovering a hidden treasure, and each clue we find reveals something wonderful about what we're learning."},
            
            # 8. CLARIFICATION & ELABORATION
            {"type": "clarification", "text": "Let me clarify that point for you - I want to make sure you have a crystal-clear understanding. Sometimes concepts can seem confusing at first, but once we break them down and look at them from different angles, everything becomes much clearer."},
            {"type": "clarification", "text": "I want to elaborate on that idea because it's such an important concept. Let me give you more details and examples so you can really see how this works in practice. The more we explore it together, the more sense it will make."},
            {"type": "clarification", "text": "Let me explain that in a different way to make sure it's completely clear. Sometimes hearing the same concept explained from a different perspective helps everything click into place. I want you to feel confident about your understanding."},
            
            # 9. VALIDATION & AFFIRMATION
            {"type": "validation", "text": "You're absolutely right about that! Your understanding is spot-on, and I'm so impressed with how you've grasped this concept. Your insights show that you're really thinking deeply about what we're learning, and that's exactly what I love to see."},
            {"type": "validation", "text": "That's such a brilliant observation! You've hit the nail on the head, and I want you to know how much I appreciate your thoughtful analysis. Your perspective adds so much value to our learning experience together."},
            {"type": "validation", "text": "I completely agree with your thinking on this! You've shown such excellent understanding, and I'm thrilled to see how well you've connected all the pieces together. Your insights are always so valuable and insightful."},
            
            # 10. TRANSITION & FLOW
            {"type": "transition", "text": "Now that we've explored this concept thoroughly, I'm excited to show you how it connects to something even more fascinating. The way these ideas build upon each other is really beautiful, and I think you're going to love what comes next."},
            {"type": "transition", "text": "Perfect! Now that you've got a solid understanding of this, let's see how it leads us to our next discovery. I love how each concept we learn opens doors to new and exciting possibilities. Are you ready for the next part of our adventure?"},
            {"type": "transition", "text": "Excellent work! Now that we've mastered this concept, I want to show you how it connects to something that will really expand your understanding. The way knowledge builds and connects is one of my favorite things about learning."},
            
            # 11. CLOSING & FAREWELL
            {"type": "closing", "text": "What an amazing learning session we've had together! I'm so proud of everything you've accomplished today, and I want you to know how much I've enjoyed our time together. You've been such a wonderful student, and I can't wait to continue our journey."},
            {"type": "closing", "text": "Before we say goodbye for today, I want you to know how much I appreciate your curiosity and dedication. You've made this such a rewarding experience for me, and I hope you feel proud of all the progress you've made. Until next time!"},
            {"type": "closing", "text": "It's been such a pleasure learning with you today! Your enthusiasm and intelligence have made this session truly special. I want you to carry with you the confidence that you're an amazing learner, and I'm excited to see what we'll discover together next time."},
            
            # 12. PERSONAL CONNECTION & RELATIONSHIP
            {"type": "personal", "text": "I want you to know how much I value our relationship and the trust you've placed in me as your teacher. Learning together has created such a special bond between us, and I feel so fortunate to be part of your educational journey. You mean a lot to me."},
            {"type": "personal", "text": "I hope you know how much I care about you and your success. Teaching you isn't just about sharing knowledge - it's about supporting you as you grow and develop into the amazing person you're becoming. I'm here for you, always."},
            {"type": "personal", "text": "Our time together means so much to me, and I want you to know that I see you not just as a student, but as a unique individual with incredible potential. Your growth and happiness are important to me, and I'm honored to be part of your learning story."},
        ]
        
        logger.info(f"Generated {len(conversation_types)} conversation type samples")
        return conversation_types
    
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
    
    def generate_comprehensive_samples(self, target_minutes: float = 60.0) -> bool:
        """Generate comprehensive training samples"""
        
        logger.info(f"Generating {target_minutes} minutes of comprehensive training data")
        
        # Generate emotional and conversation content
        emotional_content = self.generate_emotional_content()
        conversation_types = self.generate_conversation_types()
        
        # Combine all content
        all_content = emotional_content + conversation_types
        
        # Calculate target samples (assuming 10 seconds average per sample)
        target_samples = int((target_minutes * 60) / 10.0)
        
        logger.info(f"Target: {target_samples} comprehensive samples")
        logger.info(f"Content breakdown: {len(emotional_content)} emotional, {len(conversation_types)} conversation types")
        
        # Find starting index
        existing_files = list(self.audio_dir.glob("*.wav"))
        start_index = len(existing_files) + 1
        
        # Generate comprehensive audio samples
        successful_samples = 0
        failed_samples = 0
        
        # Use all content multiple times to reach target
        content_cycle = all_content * ((target_samples // len(all_content)) + 1)
        random.shuffle(content_cycle)
        
        for i, content_data in enumerate(content_cycle[:target_samples]):
            # Create unique filename
            sample_id = f"kelly25_{start_index + i:04d}"
            mp3_path = self.audio_dir / f"{sample_id}.mp3"
            wav_path = self.audio_dir / f"{sample_id}.wav"
            
            # Generate audio
            if self.generate_audio(content_data["text"], str(mp3_path)):
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
            if (i + 1) % 20 == 0:
                logger.info(f"Progress: {successful_samples} successful, {failed_samples} failed")
        
        logger.info(f"Comprehensive generation complete: {successful_samples} successful, {failed_samples} failed")
        return successful_samples > 0

def main():
    """Main function to generate comprehensive training data"""
    
    # Configuration
    API_KEY = "sk_17b7a1d5b54e992c687a165646ddf84dd3997cd748127568"
    VOICE_ID = "wAdymQH5YucAkXwmrdL0"  # Kelly25 voice
    TARGET_MINUTES = 60.0  # Generate 60 additional minutes
    
    # Initialize generator
    generator = ComprehensiveTrainingGenerator(API_KEY, VOICE_ID)
    
    # Generate comprehensive training samples
    success = generator.generate_comprehensive_samples(TARGET_MINUTES)
    
    if success:
        logger.info("✅ Comprehensive training data generation completed successfully!")
        logger.info("📝 Next steps:")
        logger.info("1. Run validation script to check total duration")
        logger.info("2. Update metadata.csv with new samples")
        logger.info("3. Validate complete 2-hour dataset")
    else:
        logger.error("❌ Comprehensive training data generation failed!")

if __name__ == "__main__":
    main()




































