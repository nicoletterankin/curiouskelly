# 🚀 ONE-CLICK SETUP: Generate All Kelly Visuals

## ✅ What's Already Done
- Supabase: **CONNECTED** (via MCP)
- Database tables: **CREATED**
- Visual contexts: **19 seeded**
- Prompt templates: **7 seeded**
- Kelly LoRA: **READY** at HuggingFace

## 🔑 What You Need to Provide: ONE TOKEN

### Get Your Replicate API Token (2 minutes)

1. **Go to:** https://replicate.com/account/api-tokens
2. **Sign in** (or create free account)
3. **Copy your API token**

### Add It to Your Environment

**Option A: Quick (Terminal)**
```powershell
# Run this in your terminal:
$env:REPLICATE_API_TOKEN = "r8_YOUR_TOKEN_HERE"
```

**Option B: Permanent (.env file)**
Add this line to your `.env` file:
```
REPLICATE_API_TOKEN=r8_YOUR_TOKEN_HERE
```

---

## 🎬 START GENERATION

Once you have the token set, run:

```powershell
# Test with Day 8 first
npm run visuals:day=8

# Then generate all missing (Days 8-365)
npm run visuals:missing
```

---

## 💰 Cost Estimate

| What | Images | Cost |
|------|--------|------|
| Days 8-365 phase visuals | 1,790 | ~$72 |
| Time | ~3 hours | Automated |

**Total: About $72 to complete ALL 365 lessons**

---

## 🎯 What Happens When You Run It

1. Fetches each lesson topic from database
2. Matches topic to visual context (environment, props, mood)
3. Generates 5 phase images per lesson (hook, q1, q2, q3, wisdom)
4. Saves to `public/kelly/phases/{day}/`
5. Records in `lesson_assets` table
6. Tracks cost in `generation_costs` table

---

## ✨ Ready?

Just paste your Replicate token and say **"Go"** - I'll handle the rest!

