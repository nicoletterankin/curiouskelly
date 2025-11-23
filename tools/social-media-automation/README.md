# Social Media Automation Tools - Curious Kelly

**Purpose:** Streamline social media posting, scheduling, analytics, and content generation.

---

## 📁 Files in This Directory

| File | Purpose |
|------|---------|
| `post_scheduler.py` | Schedule and publish posts to multiple platforms |
| `content_generator.py` | Generate social content using AI (OpenAI GPT) |
| `analytics_tracker.py` | Track performance metrics across platforms |
| `hashtag_optimizer.py` | Research and suggest optimal hashtags |
| `engagement_bot.py` | Auto-respond to comments and mentions |
| `.env.example` | Template for environment variables |
| `requirements.txt` | Python dependencies |

---

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
cd tools/social-media-automation
pip install -r requirements.txt
```

**requirements.txt:**
```
requests==2.31.0
python-dotenv==1.0.0
openai==1.3.5
tweepy==4.14.0
schedule==1.2.0
pillow==10.1.0
pandas==2.1.3
```

---

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

**.env file:**
```env
# OpenAI (for AI content generation)
OPENAI_API_KEY=sk-...

# Buffer (multi-platform scheduling)
BUFFER_ACCESS_TOKEN=your_buffer_token
BUFFER_PROFILE_IDS=twitter_id,instagram_id,linkedin_id

# Twitter / X
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_SECRET=your_access_secret

# Instagram (via Facebook Graph API)
INSTAGRAM_ACCESS_TOKEN=your_instagram_token
INSTAGRAM_BUSINESS_ACCOUNT_ID=your_account_id

# TikTok (via TikTok API)
TIKTOK_ACCESS_TOKEN=your_tiktok_token

# Discord (for community bot)
DISCORD_BOT_TOKEN=your_discord_token
DISCORD_CHANNEL_ID=your_channel_id

# Analytics
GOOGLE_ANALYTICS_KEY=your_ga_key
```

---

## 📖 Usage Examples

### Generate Content with AI

```bash
python content_generator.py
```

**What it does:**
- Generates tweets, Instagram captions, threads, video scripts
- Uses Kelly's brand voice (curious, warm, intelligent)
- Adapts tone (neutral, fun, wisdom) based on platform
- Outputs to `content/generated-content-{date}.json`

**Programmatic usage:**
```python
from content_generator import ContentGenerator

generator = ContentGenerator()

# Generate a tweet
tweet = generator.generate_twitter_post("why the sky is blue", tone="fun")
print(tweet)

# Generate Instagram caption
caption = generator.generate_instagram_caption("how plants grow", tone="fun")

# Generate Twitter thread
thread = generator.generate_thread("how memory works", num_tweets=6)

# Generate TikTok script
script = generator.generate_video_script("octopuses have 3 hearts", duration_seconds=30)
```

---

### Schedule Posts

```bash
python post_scheduler.py
```

**What it does:**
- Loads content calendar (`docs/social-media/content-calendar.json`)
- Schedules posts via Buffer API or direct platform APIs
- Tracks scheduled vs. published status
- Generates weekly analytics reports

**Programmatic usage:**
```python
from post_scheduler import SocialMediaScheduler

scheduler = SocialMediaScheduler()

# Schedule all approved posts for next week
count = scheduler.schedule_posts_for_week()

# Post immediately to Twitter
scheduler.post_to_twitter("Hello world! 🌟", media_path="image.jpg")

# Schedule specific post via Buffer
post = {
    "platform": "twitter",
    "text": "Did you know...? #CuriousKelly",
    "scheduled_time": "2025-12-17T10:00:00",
    "profile_ids": ["twitter_profile_id"]
}
scheduler.schedule_post_buffer(post)
```

---

### Track Analytics

```bash
python analytics_tracker.py
```

**What it does:**
- Fetches metrics from Buffer, Twitter, Instagram, TikTok
- Calculates engagement rates, reach, conversions
- Generates weekly/monthly reports
- Exports to CSV for further analysis

**Programmatic usage:**
```python
from analytics_tracker import AnalyticsDashboard

analytics = AnalyticsDashboard()

# Get weekly report for all platforms
profile_ids = ["twitter_id", "instagram_id", "linkedin_id"]
report = analytics.generate_weekly_report(profile_ids)
print(report)

# Get detailed metrics for specific platform
twitter_metrics = analytics.get_buffer_analytics("twitter_profile_id", days=30)
```

---

### Optimize Hashtags

```bash
python hashtag_optimizer.py --topic "science facts" --platform instagram
```

**What it does:**
- Researches trending hashtags for a topic
- Suggests mix of high-reach and niche tags
- Tracks hashtag performance over time
- Recommends optimal hashtag count per platform

**Programmatic usage:**
```python
from hashtag_optimizer import HashtagOptimizer

optimizer = HashtagOptimizer()

# Get optimal hashtags for a post
hashtags = optimizer.suggest_hashtags(
    topic="lifelong learning",
    platform="instagram",
    count=15
)

print(hashtags)
# ['#LifelongLearning', '#EdTech', '#CuriousKelly', ...]
```

---

### Engagement Bot (Auto-respond)

```bash
python engagement_bot.py --mode monitor
```

**What it does:**
- Monitors mentions, comments, DMs
- Auto-responds to common questions with templates
- Flags urgent issues for human review
- Tracks response times

**Programmatic usage:**
```python
from engagement_bot import EngagementBot

bot = EngagementBot()

# Start monitoring (runs continuously)
bot.start_monitoring()

# Respond to specific comment
bot.respond_to_comment(
    platform="twitter",
    comment_id="1234567890",
    response="Thanks for asking! Check out curiouskelly.com 🌟"
)
```

---

## 🔄 Automation Workflows

### Daily Automated Tasks

**Run via cron or Task Scheduler:**

```bash
# Every morning at 8 AM: Post today's lesson topic to Discord
0 8 * * * python engagement_bot.py --task post_daily_lesson

# Every hour: Monitor and respond to new mentions
0 * * * * python engagement_bot.py --mode monitor --duration 60

# Every night at midnight: Generate analytics report
0 0 * * * python analytics_tracker.py --report daily
```

---

### Weekly Automated Tasks

```bash
# Every Sunday at 6 PM: Generate and schedule next week's content
0 18 * * 0 python content_generator.py --generate-week && python post_scheduler.py --schedule-week

# Every Monday at 9 AM: Send weekly analytics report
0 9 * * 1 python analytics_tracker.py --report weekly --email hello@curiouskelly.com
```

---

## 📊 Content Calendar JSON Format

**Location:** `docs/social-media/content-calendar.json`

```json
{
  "posts": [
    {
      "id": "post_001",
      "date": "2025-12-17",
      "day": "Monday",
      "platform": "twitter",
      "content_type": "single_post",
      "topic": "Why the sky is blue",
      "text": "Did you know the sky is blue because...",
      "media_url": "https://example.com/image.jpg",
      "hashtags": ["#CuriousKelly", "#ScienceFacts"],
      "scheduled_time": "2025-12-17T10:00:00",
      "profile_ids": ["twitter_profile_id"],
      "status": "approved",
      "pillar": "educate",
      "tone": "fun"
    }
  ]
}
```

**Status values:**
- `idea` → `draft` → `in_review` → `approved` → `scheduled` → `published` → `analyzed`

---

## 🤖 AI Content Generation Tips

### Get Better Results

1. **Be specific with topics:** "Why octopuses have blue blood" > "Octopuses"
2. **Choose appropriate tone:**
   - `neutral` for LinkedIn, announcements
   - `fun` for TikTok, Instagram, Twitter
   - `wisdom` for inspirational quotes, Sunday posts
3. **Test and iterate:** Generate 3 variations, pick the best
4. **Human review:** AI is great, but humans approve final posts
5. **Brand consistency:** Always review against brand guidelines

### Prompt Engineering

The `content_generator.py` uses carefully crafted system prompts that include:
- Kelly's personality traits
- Platform-specific constraints
- Brand voice characteristics
- Formatting requirements

You can customize these in the `ContentGenerator` class.

---

## 🔒 Security Best Practices

1. **Never commit `.env` file** (already in `.gitignore`)
2. **Rotate API keys quarterly**
3. **Use read-only tokens** when possible
4. **Review auto-generated content** before publishing
5. **Set rate limits** to avoid API throttling
6. **Monitor for unauthorized access**

---

## 📈 Performance Optimization

### Batch Operations

Instead of generating/scheduling one post at a time:

```python
# Generate a week of content in one batch
topics = ["Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5", "Topic 6", "Topic 7"]
weekly_content = generator.batch_generate_week_content(topics, start_date)

# Schedule all at once
scheduler.schedule_posts_for_week(start_date)
```

### Caching

Content generator caches common requests:

```python
# First call: API request (slow)
tweet1 = generator.generate_twitter_post("sky is blue")

# Second call: Cached result (fast)
tweet2 = generator.generate_twitter_post("sky is blue")
```

---

## 🛠️ Troubleshooting

### Common Issues

**1. "API key not found" error**
- Solution: Check `.env` file is in same directory as script
- Verify key name matches exactly (e.g., `OPENAI_API_KEY` not `OPENAI_KEY`)

**2. "Rate limit exceeded"**
- Solution: Add delays between requests
- Use batch operations
- Upgrade API plan if needed

**3. "Content generation fails"**
- Solution: Check OpenAI API balance
- Verify model name is correct (`gpt-4` or `gpt-3.5-turbo`)
- Review API logs for specific error

**4. "Posts not scheduling"**
- Solution: Verify Buffer profile IDs are correct
- Check post status is `approved` in calendar
- Ensure scheduled_time is in future

---

## 📚 Additional Resources

- **Buffer API Docs:** https://buffer.com/developers/api
- **Twitter API Docs:** https://developer.twitter.com/en/docs
- **OpenAI API Docs:** https://platform.openai.com/docs
- **Instagram Graph API:** https://developers.facebook.com/docs/instagram-api

---

## ✅ Automation Checklist

**Before launching automation:**
- [ ] All API keys added to `.env`
- [ ] Content calendar populated with at least 2 weeks of posts
- [ ] Brand guidelines reviewed
- [ ] Test posts published manually to verify formatting
- [ ] Analytics tracking confirmed working
- [ ] Team trained on approval workflow
- [ ] Cron jobs / scheduled tasks configured
- [ ] Monitoring alerts set up
- [ ] Backup system in place (if automation fails)

---

## 🤝 Contributing

To add new automation features:

1. Create new Python script in this directory
2. Follow naming convention: `{feature}_automation.py`
3. Add to this README with usage examples
4. Update `requirements.txt` if new dependencies needed
5. Test thoroughly before deploying to production

---

## 📞 Support

**Questions about automation?**
- Check this README first
- Review code comments in Python files
- Ask in Discord #tech-help channel
- Email: hello@curiouskelly.com

---

**Last Updated:** November 21, 2025  
**Maintained by:** Technical Lead

🤖 **Happy automating! Work smarter, not harder.**

