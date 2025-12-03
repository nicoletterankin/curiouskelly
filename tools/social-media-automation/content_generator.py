"""
Curious Kelly - AI Content Generator
Uses OpenAI GPT to generate social media content based on lesson topics

Requirements:
    pip install openai python-dotenv

Environment Variables (.env):
    OPENAI_API_KEY=your_openai_key
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ContentGenerator:
    """Generates social media content using AI"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            print("⚠️ OPENAI_API_KEY not found. Set in .env file.")
        
        # Kelly's brand voice characteristics
        self.brand_voice = {
            "personality": ["curious", "warm", "intelligent", "enthusiastic", "inclusive"],
            "tone_modes": {
                "neutral": "professional, informative, data-driven",
                "fun": "playful, energetic, emoji-friendly",
                "wisdom": "reflective, thoughtful, poetic"
            },
            "target_audiences": ["adults 25-65", "children 2-17 via parents", "teachers/educators"]
        }
    
    def generate_twitter_post(
        self, 
        topic: str, 
        tone: str = "fun",
        include_cta: bool = True
    ) -> str:
        """
        Generate a Twitter post about a topic
        
        Args:
            topic: Lesson topic or fact to post about
            tone: Voice mode ('neutral', 'fun', or 'wisdom')
            include_cta: Whether to include call-to-action
        
        Returns:
            str: Generated tweet (max 280 characters)
        """
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            system_prompt = f"""You are Kelly, an AI learning companion for Curious Kelly. 
Your voice is {self.brand_voice['tone_modes'].get(tone, 'fun')}.

Brand characteristics: {', '.join(self.brand_voice['personality'])}

Generate a Twitter post (max 280 characters) about: {topic}

Requirements:
- Start with a hook (surprising fact or question)
- Be educational but entertaining
- Use 1-2 emojis maximum
- Include #CuriousKelly hashtag
- {'Include subtle CTA: "Learn more: curiouskelly.com"' if include_cta else 'No CTA needed'}
- Sound like a human, not a corporate account
"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Create a tweet about: {topic}"}
                ],
                max_tokens=100,
                temperature=0.8
            )
            
            tweet = response.choices[0].message.content.strip()
            
            # Ensure under 280 chars
            if len(tweet) > 280:
                tweet = tweet[:277] + "..."
            
            return tweet
            
        except Exception as e:
            print(f"❌ Error generating tweet: {e}")
            return f"Did you know about {topic}? Learn more at curiouskelly.com #CuriousKelly"
    
    def generate_instagram_caption(
        self,
        topic: str,
        tone: str = "fun",
        include_hashtags: bool = True
    ) -> str:
        """
        Generate an Instagram caption
        
        Args:
            topic: Lesson topic
            tone: Voice mode
            include_hashtags: Whether to include hashtag block
        
        Returns:
            str: Instagram caption (optimized for engagement)
        """
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            system_prompt = f"""You are Kelly, an AI learning companion for Curious Kelly.
Your voice is {self.brand_voice['tone_modes'].get(tone, 'fun')}.

Generate an Instagram caption about: {topic}

Structure:
1. Hook (1 line that grabs attention)
2. Story or explanation (2-3 paragraphs with line breaks)
3. Engagement question ("What do you think?")
4. Subtle CTA ("Link in bio for daily lessons 🌟")
{'5. Hashtag block (10-15 relevant hashtags)' if include_hashtags else ''}

Style:
- Use emojis strategically (2-4 total)
- Line breaks for readability
- Conversational but knowledgeable
- Encourage comments
"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Create an Instagram caption about: {topic}"}
                ],
                max_tokens=400,
                temperature=0.8
            )
            
            caption = response.choices[0].message.content.strip()
            
            # Add default hashtags if not included
            if include_hashtags and '#CuriousKelly' not in caption:
                caption += "\n\n#CuriousKelly #DailyLesson #LifelongLearning #Education"
            
            return caption
            
        except Exception as e:
            print(f"❌ Error generating Instagram caption: {e}")
            return f"Today's lesson: {topic}\n\nLearn something new every day with Curious Kelly.\nLink in bio 🌟\n\n#CuriousKelly #DailyLesson"
    
    def generate_thread(
        self,
        topic: str,
        num_tweets: int = 5
    ) -> List[str]:
        """
        Generate a Twitter thread (educational deep-dive)
        
        Args:
            topic: Topic to create thread about
            num_tweets: Number of tweets in thread
        
        Returns:
            List[str]: List of tweets forming the thread
        """
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            system_prompt = f"""You are Kelly, an AI learning companion for Curious Kelly.

Create a {num_tweets}-tweet thread about: {topic}

Thread structure:
1. Hook tweet (surprising fact, bold claim)
2-{num_tweets-2}. Educational content (break down the topic)
{num_tweets-1}. Key takeaway
{num_tweets}. CTA (link to curiouskelly.com)

Rules:
- Each tweet max 280 characters
- Number tweets (1/{num_tweets}, 2/{num_tweets}, etc.)
- Use 🧵 emoji in first tweet
- Be educational but entertaining
- End with #CuriousKelly
"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Create a thread about: {topic}"}
                ],
                max_tokens=500,
                temperature=0.8
            )
            
            thread_text = response.choices[0].message.content.strip()
            
            # Parse into individual tweets
            tweets = []
            for line in thread_text.split('\n\n'):
                if line.strip():
                    # Ensure under 280 chars
                    tweet = line.strip()
                    if len(tweet) > 280:
                        tweet = tweet[:277] + "..."
                    tweets.append(tweet)
            
            return tweets[:num_tweets]
            
        except Exception as e:
            print(f"❌ Error generating thread: {e}")
            return [
                f"Let's talk about {topic} 🧵👇 (1/{num_tweets})",
                f"[Tweet 2 about {topic}] (2/{num_tweets})",
                f"[Tweet 3 about {topic}] (3/{num_tweets})",
                f"Key takeaway: {topic} is fascinating! (4/{num_tweets})",
                f"Want to learn more? Visit curiouskelly.com #CuriousKelly ({num_tweets}/{num_tweets})"
            ]
    
    def generate_video_script(
        self,
        topic: str,
        duration_seconds: int = 30,
        platform: str = "tiktok"
    ) -> Dict[str, str]:
        """
        Generate a video script for TikTok/Reels
        
        Args:
            topic: Video topic
            duration_seconds: Target video length
            platform: Platform (tiktok, instagram, youtube)
        
        Returns:
            Dict with 'script', 'hook', 'visual_notes'
        """
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            system_prompt = f"""You are Kelly, creating a {duration_seconds}-second video script for {platform}.

Topic: {topic}

Script structure:
[0-3s] HOOK: (Surprising statement or question to stop scroll)
[4-{duration_seconds-10}s] EXPLANATION: (Break down the topic simply)
[{duration_seconds-9}-{duration_seconds-3}s] PAYOFF: (Interesting conclusion or twist)
[{duration_seconds-2}-{duration_seconds}s] CTA: (Follow @CuriousKellyAI 🌟)

Include:
- Voiceover script (what Kelly says)
- Visual notes (what viewers see)
- On-screen text suggestions
- Energy level (high, medium, calm)

Keep it:
- Fast-paced and engaging
- Educational but entertaining
- Suitable for all ages
"""
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Create a video script about: {topic}"}
                ],
                max_tokens=600,
                temperature=0.8
            )
            
            script = response.choices[0].message.content.strip()
            
            # Parse script into components
            return {
                "full_script": script,
                "topic": topic,
                "duration": duration_seconds,
                "platform": platform
            }
            
        except Exception as e:
            print(f"❌ Error generating video script: {e}")
            return {
                "full_script": f"[0-3s] Did you know about {topic}? [4-27s] Here's why it's cool... [28-30s] Follow for more! 🌟",
                "topic": topic,
                "duration": duration_seconds,
                "platform": platform
            }
    
    def batch_generate_week_content(
        self,
        topics: List[str],
        start_date: datetime
    ) -> Dict[str, List[Dict]]:
        """
        Generate a week's worth of content for multiple platforms
        
        Args:
            topics: List of daily topics (7 topics for 7 days)
            start_date: Monday of the week
        
        Returns:
            Dict with platform keys containing lists of posts
        """
        weekly_content = {
            "twitter": [],
            "instagram": [],
            "tiktok": [],
            "linkedin": []
        }
        
        print(f"🔄 Generating content for week of {start_date.strftime('%Y-%m-%d')}...")
        
        for i, topic in enumerate(topics[:7]):
            post_date = start_date + timedelta(days=i)
            day_name = post_date.strftime('%A')
            
            print(f"   📅 {day_name}: {topic}")
            
            # Generate Twitter post
            twitter_post = self.generate_twitter_post(topic, tone="fun")
            weekly_content["twitter"].append({
                "date": post_date.isoformat(),
                "day": day_name,
                "topic": topic,
                "content": twitter_post,
                "scheduled_time": f"{post_date.strftime('%Y-%m-%d')} 10:00:00"
            })
            
            # Generate Instagram caption
            instagram_caption = self.generate_instagram_caption(topic, tone="fun")
            weekly_content["instagram"].append({
                "date": post_date.isoformat(),
                "day": day_name,
                "topic": topic,
                "caption": instagram_caption,
                "scheduled_time": f"{post_date.strftime('%Y-%m-%d')} 12:00:00"
            })
            
            # Generate TikTok script (MWF only)
            if i in [0, 2, 4]:  # Monday, Wednesday, Friday
                tiktok_script = self.generate_video_script(topic, duration_seconds=30, platform="tiktok")
                weekly_content["tiktok"].append({
                    "date": post_date.isoformat(),
                    "day": day_name,
                    "topic": topic,
                    "script": tiktok_script,
                    "scheduled_time": f"{post_date.strftime('%Y-%m-%d')} 18:00:00"
                })
        
        # Save to file
        output_path = f"content/generated-content-{start_date.strftime('%Y%m%d')}.json"
        os.makedirs('content', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(weekly_content, f, indent=2)
        
        print(f"✅ Content saved to {output_path}")
        return weekly_content


def main():
    """Example usage"""
    print("🤖 Curious Kelly - AI Content Generator")
    print("=" * 50)
    
    generator = ContentGenerator()
    
    # Example 1: Generate a single tweet
    print("\n1️⃣ Generating Twitter post...")
    tweet = generator.generate_twitter_post("why the sky is blue", tone="fun")
    print(f"Tweet: {tweet}\n")
    
    # Example 2: Generate Instagram caption
    print("2️⃣ Generating Instagram caption...")
    caption = generator.generate_instagram_caption("why leaves change color", tone="fun")
    print(f"Caption:\n{caption}\n")
    
    # Example 3: Generate thread
    print("3️⃣ Generating Twitter thread...")
    thread = generator.generate_thread("how memory works", num_tweets=5)
    for i, tweet in enumerate(thread, 1):
        print(f"Tweet {i}: {tweet}")
    
    # Example 4: Generate video script
    print("\n4️⃣ Generating TikTok script...")
    script = generator.generate_video_script("octopuses have three hearts", duration_seconds=30)
    print(f"Script:\n{script['full_script']}")
    
    # Example 5: Batch generate week content
    print("\n5️⃣ Generating week of content...")
    topics = [
        "Why do we dream?",
        "How rainbows form",
        "What makes music sound good?",
        "Why the ocean is salty",
        "How plants breathe",
        "What causes thunder",
        "Why we yawn"
    ]
    
    from datetime import datetime, timedelta
    today = datetime.now()
    next_monday = today + timedelta(days=(7 - today.weekday()))
    
    # Uncomment to generate full week:
    # weekly_content = generator.batch_generate_week_content(topics, next_monday)
    
    print("\n✅ Content generation complete!")


if __name__ == "__main__":
    main()

















