"""
Quick Post Script - Post to Twitter immediately
Use this for manual posting while setting up full automation

Usage:
    python quick_post.py                    # Post first approved post from calendar
    python quick_post.py --text "Hello!"    # Post custom text
    python quick_post.py --test             # Test connection without posting
"""

import os
import sys
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))


def get_twitter_client():
    """Get authenticated Twitter client"""
    try:
        import tweepy
    except ImportError:
        print("❌ tweepy not installed. Run: pip install tweepy")
        return None
    
    api_key = os.getenv('TWITTER_API_KEY')
    api_secret = os.getenv('TWITTER_API_SECRET')
    access_token = os.getenv('TWITTER_ACCESS_TOKEN')
    access_secret = os.getenv('TWITTER_ACCESS_SECRET')
    
    if not all([api_key, api_secret, access_token, access_secret]):
        print("❌ Twitter API credentials not configured in .env")
        print("   Required: TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET")
        return None
    
    try:
        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_secret)
        api = tweepy.API(auth)
        
        # Verify credentials
        api.verify_credentials()
        return api
    except Exception as e:
        print(f"❌ Twitter authentication failed: {e}")
        return None


def post_to_twitter(text, media_path=None):
    """Post a tweet"""
    api = get_twitter_client()
    if not api:
        return False
    
    try:
        if media_path and os.path.exists(media_path):
            media = api.media_upload(media_path)
            tweet = api.update_status(text, media_ids=[media.media_id])
        else:
            tweet = api.update_status(text)
        
        print(f"✅ Posted to Twitter!")
        print(f"   Tweet ID: {tweet.id}")
        print(f"   URL: https://twitter.com/CuriousKelly/status/{tweet.id}")
        return True
    except Exception as e:
        print(f"❌ Failed to post: {e}")
        return False


def get_next_approved_post():
    """Get next approved post from content calendar"""
    calendar_path = os.path.join(project_root, 'docs', 'social-media', 'content-calendar.json')
    
    try:
        with open(calendar_path, 'r', encoding='utf-8') as f:
            calendar = json.load(f)
        
        for post in calendar.get('posts', []):
            if post.get('status') == 'approved' and post.get('platform') == 'twitter':
                return post
        
        print("⚠️ No approved Twitter posts found in calendar")
        return None
    except FileNotFoundError:
        print(f"❌ Calendar not found: {calendar_path}")
        return None


def test_connection():
    """Test Twitter API connection"""
    print("🧪 Testing Twitter API connection...")
    api = get_twitter_client()
    
    if api:
        user = api.verify_credentials()
        print(f"✅ Connected as @{user.screen_name}")
        print(f"   Followers: {user.followers_count}")
        print(f"   Following: {user.friends_count}")
        print(f"   Tweets: {user.statuses_count}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Quick post to Twitter')
    parser.add_argument('--text', type=str, help='Custom text to post')
    parser.add_argument('--media', type=str, help='Path to media file')
    parser.add_argument('--test', action='store_true', help='Test connection only')
    parser.add_argument('--show-next', action='store_true', help='Show next scheduled post')
    args = parser.parse_args()
    
    print("=" * 50)
    print("🌟 Curious Kelly - Quick Post")
    print("=" * 50)
    
    if args.test:
        test_connection()
        return
    
    if args.show_next:
        post = get_next_approved_post()
        if post:
            print(f"\n📅 Next approved post:")
            print(f"   Date: {post.get('date')}")
            print(f"   Topic: {post.get('topic')}")
            print(f"   Text:\n{post.get('text')}")
        return
    
    if args.text:
        # Post custom text
        print(f"\n📝 Posting custom text...")
        post_to_twitter(args.text, args.media)
    else:
        # Post next from calendar
        post = get_next_approved_post()
        if post:
            print(f"\n📅 Posting from calendar:")
            print(f"   Topic: {post.get('topic')}")
            print(f"   Text preview: {post.get('text', '')[:100]}...")
            
            confirm = input("\n   Post this? (y/n): ").lower().strip()
            if confirm == 'y':
                post_to_twitter(post.get('text', ''))
            else:
                print("   Cancelled.")
        else:
            print("\n💡 Usage:")
            print("   python quick_post.py --text 'Your tweet here'")
            print("   python quick_post.py --test")
            print("   python quick_post.py --show-next")


if __name__ == "__main__":
    main()



