"""
Curious Kelly - Social Media Post Scheduler
Automates posting to multiple platforms using Buffer API or direct platform APIs

Requirements:
    pip install requests python-dotenv schedule tweepy instagrapi pillow

Environment Variables (.env):
    BUFFER_ACCESS_TOKEN=your_buffer_token
    TWITTER_API_KEY=your_twitter_key
    TWITTER_API_SECRET=your_twitter_secret
    TWITTER_ACCESS_TOKEN=your_access_token
    TWITTER_ACCESS_SECRET=your_access_secret
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv
import schedule
import time

# Load environment variables
load_dotenv()

class SocialMediaScheduler:
    """Schedules and posts content to social media platforms"""
    
    def __init__(self):
        self.buffer_token = os.getenv('BUFFER_ACCESS_TOKEN')
        self.twitter_api_key = os.getenv('TWITTER_API_KEY')
        self.content_calendar_path = 'docs/social-media/content-calendar.json'
        
    def load_content_calendar(self) -> List[Dict]:
        """Load posts from content calendar JSON"""
        try:
            with open(self.content_calendar_path, 'r', encoding='utf-8') as f:
                calendar = json.load(f)
            return calendar.get('posts', [])
        except FileNotFoundError:
            print(f"❌ Content calendar not found at {self.content_calendar_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing content calendar: {e}")
            return []
    
    def schedule_post_buffer(self, post: Dict) -> bool:
        """
        Schedule a post using Buffer API
        
        Args:
            post: Dict with keys: platform, text, media_url, scheduled_time, profile_ids
        
        Returns:
            bool: Success status
        """
        if not self.buffer_token:
            print("❌ Buffer access token not found")
            return False
        
        url = "https://api.bufferapp.com/1/updates/create.json"
        
        payload = {
            "access_token": self.buffer_token,
            "profile_ids": post.get('profile_ids', []),
            "text": post.get('text', ''),
            "scheduled_at": post.get('scheduled_time'),
            "shorten": True,
        }
        
        # Add media if provided
        if post.get('media_url'):
            payload['media'] = {
                'photo': post['media_url']
            }
        
        try:
            response = requests.post(url, data=payload)
            response.raise_for_status()
            
            result = response.json()
            print(f"✅ Post scheduled via Buffer: {result.get('id')}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Buffer API error: {e}")
            return False
    
    def post_to_twitter(self, text: str, media_path: Optional[str] = None) -> bool:
        """
        Post directly to Twitter using tweepy
        
        Args:
            text: Tweet text (280 char max)
            media_path: Optional path to image file
        
        Returns:
            bool: Success status
        """
        try:
            import tweepy
            
            # Authenticate
            auth = tweepy.OAuthHandler(
                os.getenv('TWITTER_API_KEY'),
                os.getenv('TWITTER_API_SECRET')
            )
            auth.set_access_token(
                os.getenv('TWITTER_ACCESS_TOKEN'),
                os.getenv('TWITTER_ACCESS_SECRET')
            )
            api = tweepy.API(auth)
            
            # Post tweet
            if media_path and os.path.exists(media_path):
                media = api.media_upload(media_path)
                tweet = api.update_status(text, media_ids=[media.media_id])
            else:
                tweet = api.update_status(text)
            
            print(f"✅ Posted to Twitter: {tweet.id}")
            return True
            
        except Exception as e:
            print(f"❌ Twitter post error: {e}")
            return False
    
    def schedule_posts_for_week(self, start_date: Optional[datetime] = None) -> int:
        """
        Schedule all posts for the upcoming week
        
        Args:
            start_date: Start of week (defaults to next Monday)
        
        Returns:
            int: Number of posts scheduled
        """
        if not start_date:
            # Default to next Monday
            today = datetime.now()
            days_until_monday = (7 - today.weekday()) % 7
            start_date = today + timedelta(days=days_until_monday)
        
        end_date = start_date + timedelta(days=7)
        
        posts = self.load_content_calendar()
        scheduled_count = 0
        
        for post in posts:
            post_date = datetime.fromisoformat(post.get('scheduled_time', ''))
            
            # Check if post is in target week
            if start_date <= post_date < end_date:
                if post.get('status') == 'approved':
                    success = self.schedule_post_buffer(post)
                    if success:
                        scheduled_count += 1
                        post['status'] = 'scheduled'
        
        # Save updated calendar
        self.save_content_calendar(posts)
        
        print(f"✅ Scheduled {scheduled_count} posts for {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
        return scheduled_count
    
    def save_content_calendar(self, posts: List[Dict]):
        """Save updated content calendar"""
        try:
            with open(self.content_calendar_path, 'w', encoding='utf-8') as f:
                json.dump({'posts': posts}, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving content calendar: {e}")
    
    def generate_daily_post(self, topic: str, target_platform: str = 'twitter') -> str:
        """
        Generate a daily lesson post for a topic
        
        Args:
            topic: Lesson topic
            target_platform: Platform to optimize for
        
        Returns:
            str: Generated post text
        """
        templates = {
            'twitter': f"Today's lesson: {topic}\n\n[Interesting fact about {topic}]\n\nLearn more: https://curiouskelly.com\n\n#CuriousKelly #DailyLesson",
            'instagram': f"📚 Today's Lesson: {topic}\n\n[Engaging explanation]\n\nSwipe to learn more →\n\n#CuriousKelly #DailyLesson #Learning",
            'linkedin': f"Today's Daily Lesson explores {topic}.\n\n[Professional explanation]\n\nRead more: https://curiouskelly.com\n\n#EdTech #LifelongLearning"
        }
        
        return templates.get(target_platform, templates['twitter'])


class AnalyticsDashboard:
    """Fetches and analyzes social media analytics"""
    
    def __init__(self):
        self.buffer_token = os.getenv('BUFFER_ACCESS_TOKEN')
    
    def get_buffer_analytics(self, profile_id: str, days: int = 7) -> Dict:
        """
        Fetch analytics from Buffer for a profile
        
        Args:
            profile_id: Buffer profile ID
            days: Number of days to analyze
        
        Returns:
            dict: Analytics data
        """
        if not self.buffer_token:
            print("❌ Buffer access token not found")
            return {}
        
        url = f"https://api.bufferapp.com/1/profiles/{profile_id}/updates/sent.json"
        
        params = {
            "access_token": self.buffer_token,
            "count": 100
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            updates = response.json().get('updates', [])
            
            # Calculate metrics
            total_likes = sum(u.get('statistics', {}).get('likes', 0) for u in updates)
            total_comments = sum(u.get('statistics', {}).get('comments', 0) for u in updates)
            total_shares = sum(u.get('statistics', {}).get('shares', 0) for u in updates)
            total_clicks = sum(u.get('statistics', {}).get('clicks', 0) for u in updates)
            
            return {
                'profile_id': profile_id,
                'posts_analyzed': len(updates),
                'total_likes': total_likes,
                'total_comments': total_comments,
                'total_shares': total_shares,
                'total_clicks': total_clicks,
                'avg_engagement': (total_likes + total_comments + total_shares) / len(updates) if updates else 0
            }
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Buffer API error: {e}")
            return {}
    
    def generate_weekly_report(self, profile_ids: List[str]) -> str:
        """
        Generate a weekly analytics report
        
        Args:
            profile_ids: List of Buffer profile IDs to analyze
        
        Returns:
            str: Formatted report
        """
        report_lines = ["📊 Weekly Social Media Report\n", "=" * 50, ""]
        
        for profile_id in profile_ids:
            analytics = self.get_buffer_analytics(profile_id, days=7)
            
            if analytics:
                report_lines.extend([
                    f"\n🔹 Profile: {profile_id}",
                    f"   Posts: {analytics['posts_analyzed']}",
                    f"   Likes: {analytics['total_likes']}",
                    f"   Comments: {analytics['total_comments']}",
                    f"   Shares: {analytics['total_shares']}",
                    f"   Clicks: {analytics['total_clicks']}",
                    f"   Avg Engagement: {analytics['avg_engagement']:.1f}",
                ])
        
        report_lines.extend(["", "=" * 50])
        report = "\n".join(report_lines)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"metrics/social-media-report-{timestamp}.txt"
        
        os.makedirs('metrics', exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Report saved to {report_path}")
        return report


def main():
    """Main execution function"""
    print("🌟 Curious Kelly - Social Media Automation")
    print("=" * 50)
    
    scheduler = SocialMediaScheduler()
    analytics = AnalyticsDashboard()
    
    # Example usage:
    
    # 1. Schedule posts for next week
    print("\n1️⃣ Scheduling posts for next week...")
    count = scheduler.schedule_posts_for_week()
    
    # 2. Generate analytics report
    print("\n2️⃣ Generating analytics report...")
    profile_ids = os.getenv('BUFFER_PROFILE_IDS', '').split(',')
    if profile_ids:
        report = analytics.generate_weekly_report(profile_ids)
        print(report)
    
    # 3. Post immediate content (example)
    # scheduler.post_to_twitter("Hello from Curious Kelly! 🌟 #DailyLesson")
    
    print("\n✅ Automation complete!")


if __name__ == "__main__":
    main()
































