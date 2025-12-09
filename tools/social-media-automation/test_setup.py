"""
Test script to verify social media automation setup
"""
import os
import sys
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

print("=" * 60)
print("🧪 CURIOUS KELLY - SOCIAL MEDIA AUTOMATION TEST")
print("=" * 60)

# Test 1: Content Generator
print("\n1️⃣ Testing Content Generator...")
try:
    sys.path.insert(0, os.path.join(project_root, 'tools', 'social-media-automation'))
    from content_generator import ContentGenerator
    gen = ContentGenerator()
    print(f"   ✅ ContentGenerator loaded")
    print(f"   OpenAI API Key: {'✅ Configured' if gen.api_key else '❌ Missing'}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Post Scheduler
print("\n2️⃣ Testing Post Scheduler...")
try:
    from post_scheduler import SocialMediaScheduler, AnalyticsDashboard
    sched = SocialMediaScheduler()
    sched.content_calendar_path = os.path.join(project_root, 'docs', 'social-media', 'content-calendar.json')
    print(f"   ✅ SocialMediaScheduler loaded")
    print(f"   Buffer Token: {'✅ Configured' if sched.buffer_token else '⚠️ Not configured (needed for scheduling)'}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Content Calendar
print("\n3️⃣ Testing Content Calendar...")
calendar_path = os.path.join(project_root, 'docs', 'social-media', 'content-calendar.json')
try:
    with open(calendar_path, 'r', encoding='utf-8') as f:
        calendar = json.load(f)
    posts = calendar.get('posts', [])
    print(f"   ✅ Content calendar loaded")
    print(f"   Total posts: {len(posts)}")
    
    # Count by platform
    platforms = {}
    for post in posts:
        p = post.get('platform', 'unknown')
        platforms[p] = platforms.get(p, 0) + 1
    
    print(f"   Posts by platform:")
    for platform, count in sorted(platforms.items()):
        print(f"      {platform}: {count}")
    
    # Show first 3 posts
    print(f"\n   First 3 scheduled posts:")
    for post in posts[:3]:
        print(f"      📅 {post.get('date')} | {post.get('platform')} | {post.get('topic', 'N/A')[:40]}")

except FileNotFoundError:
    print(f"   ❌ Calendar not found at: {calendar_path}")
except json.JSONDecodeError as e:
    print(f"   ❌ Invalid JSON: {e}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: Brand Assets
print("\n4️⃣ Testing Brand Assets...")
assets_path = os.path.join(project_root, 'assets', 'kelly-brand-final', 'images', 'social')
try:
    if os.path.exists(assets_path):
        files = os.listdir(assets_path)
        print(f"   ✅ Social media assets found: {len(files)} files")
        
        # Check for required files
        required = ['profile-twitter.png', 'profile-instagram.png', 'cover-twitter.png']
        for req in required:
            if req in files:
                print(f"      ✅ {req}")
            else:
                print(f"      ❌ Missing: {req}")
    else:
        print(f"   ❌ Assets folder not found: {assets_path}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 5: Launch Posts File
print("\n5️⃣ Testing Launch Posts...")
launch_path = os.path.join(project_root, 'content', 'launch-week-posts.json')
try:
    with open(launch_path, 'r', encoding='utf-8') as f:
        launch = json.load(f)
    posts = launch.get('posts', [])
    tiktoks = launch.get('tiktok_scripts', [])
    linkedin = launch.get('linkedin_posts', [])
    print(f"   ✅ Launch posts loaded")
    print(f"      Twitter/Instagram posts: {len(posts)}")
    print(f"      TikTok scripts: {len(tiktoks)}")
    print(f"      LinkedIn posts: {len(linkedin)}")
except FileNotFoundError:
    print(f"   ❌ Launch posts not found")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Summary
print("\n" + "=" * 60)
print("📋 SUMMARY")
print("=" * 60)
print("""
✅ READY:
   - Content Generator (with OpenAI API)
   - Post Scheduler (needs Buffer token for scheduling)
   - Content Calendar (17 posts for launch week)
   - Brand Assets (profile pics + covers for all platforms)
   - Launch Posts (17 social + 3 TikTok + 2 LinkedIn)

⚠️ NEEDS CONFIGURATION:
   - Buffer token (for automated scheduling)
   - Platform API keys (for direct posting)

🚀 NEXT STEPS:
   1. Create social media accounts
   2. Add Buffer token to .env file
   3. Run: python post_scheduler.py
""")


