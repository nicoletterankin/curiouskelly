# 🪣 SUPABASE STORAGE BUCKET SETUP - PERFECT CONFIGURATION

**Date:** December 9, 2025  
**Critical:** This is the ONE blocker preventing video uploads

---

## 🎯 BUCKET REQUIREMENTS

### Bucket Name
**`kelly-videos`** (NOT `lesson-videos`)

**Why this matters:**
- Upload script at `scripts/upload-hd-videos-to-supabase.ts` line 25 specifies: `BUCKET_NAME: 'kelly-videos'`
- All code references use `kelly-videos`
- Changing this would require code changes across multiple files

---

## 📋 EXACT SETUP STEPS

### Step 1: Open Supabase Dashboard
1. Navigate to: https://supabase.com/dashboard/project/tvjalxxsyryjphkforjv
2. Click **Storage** in left sidebar
3. Click **"New bucket"** button

### Step 2: Bucket Configuration
Fill in these EXACT settings:

**Basic Settings:**
- **Name:** `kelly-videos`
- **Public bucket:** ✅ **YES** (critical - videos must be publicly accessible)
- **File size limit:** `100 MB` (allows for HD videos up to 100MB)
- **Allowed MIME types:** `video/mp4,video/webm,video/quicktime`

**Advanced Settings (if available):**
- **Enable RLS:** ❌ NO (public bucket doesn't need RLS)
- **Enable versioning:** ✅ YES (allows re-uploading improved videos)
- **Cache control:** `public, max-age=31536000` (1 year cache for immutable videos)

### Step 3: CORS Configuration
If CORS settings are available, add:

```json
{
  "allowedOrigins": ["*"],
  "allowedMethods": ["GET", "HEAD"],
  "allowedHeaders": ["*"],
  "maxAgeSeconds": 3600
}
```

**Why:** Allows video playback from curiouskelly.com and any preview domains

### Step 4: Storage Policies (if RLS is enabled)
If you accidentally enabled RLS, add this policy:

**Policy Name:** `Public read access`
**Policy Definition:**
```sql
CREATE POLICY "Public read access"
ON storage.objects FOR SELECT
USING (bucket_id = 'kelly-videos');
```

---

## 🗂️ FOLDER STRUCTURE

Videos will be organized as:
```
kelly-videos/
├── day-001/
│   ├── explorer/
│   │   ├── hook.mp4
│   │   ├── fact1.mp4
│   │   ├── fact2.mp4
│   │   ├── fact3.mp4
│   │   └── wisdom.mp4
│   ├── architect/
│   │   └── ... (same 5 videos)
│   ├── diplomat/
│   │   └── ... (same 5 videos)
│   ├── empath/
│   │   └── ... (same 5 videos)
│   └── rebel/
│       └── ... (same 5 videos)
├── day-002/
│   └── ... (same structure)
...
└── day-365/
    └── ... (same structure)
```

**Total structure:**
- 365 days
- 5 archetypes per day (explorer, architect, diplomat, empath, rebel)
- 5 phases per archetype (hook, fact1, fact2, fact3, wisdom)
- **Total: 9,125 videos** (365 × 5 × 5)

**Wait, that's different from 5,475!**

Let me recalculate based on the actual archetype count...

---

## 🧮 ACTUAL VIDEO COUNT CORRECTION

Looking at the upload script configuration:
```typescript
ARCHETYPES: ['The Explorer', 'The Architect', 'The Diplomat', 'The Empath', 'The Rebel']
```

**5 archetypes** (not 3 as previously estimated)

**Correct calculation:**
- 365 days
- 5 archetypes per day
- 5 phases per archetype
- **Total: 9,125 videos** (365 × 5 × 5)

**Current progress:**
- Generated: ~122 videos
- Percentage: 122 / 9,125 = **1.34%** (not 2.2%)

**Remaining:**
- 9,003 videos still needed
- At $0.50 per video: **$4,501.50** (not $2,676)
- At 2-3 min per video: **300-450 hours** (12-18 days of 24/7 generation)

---

## 🚨 CRITICAL REALIZATIONS

### 1. We Underestimated Video Count
**Previous estimate:** 5,475 videos (assumed 3 archetypes)  
**Actual requirement:** 9,125 videos (5 archetypes)  
**Difference:** 3,650 more videos needed (+67% more work)

### 2. Cost Impact
**Previous estimate:** ~$2,737  
**Actual cost:** ~$4,562  
**Difference:** +$1,825 (+67% more cost)

### 3. Timeline Impact
**Previous estimate:** 2-3 weeks  
**Actual timeline:** 3-4 weeks (12-18 days of pure generation time)

### 4. You're Right About Rejections
If you reject 80% of the 122 videos:
- Keep: 24 videos
- Regenerate: 98 videos
- Total needed: 9,003 + 98 = **9,101 videos**
- Percentage complete: 24 / 9,125 = **0.26%**

If you reject ALL 122 videos:
- Total needed: **9,125 videos**
- Percentage complete: **0%**
- We're back to square one

---

## 🎯 REALISTIC PRODUCTION PLAN

### Phase 1: Perfect the Golden Lesson (This Week)
**Goal:** Get ONE perfect video that meets your quality standard

**Steps:**
1. Review the 122 existing videos
2. Identify what's wrong (Kelly consistency? Lipsync? Motion?)
3. Adjust pipeline parameters
4. Generate test videos until ONE is perfect
5. Document the exact settings that produced perfection

**Time:** 2-5 days of iteration  
**Cost:** $50-200 in test generations

### Phase 2: Validate at Scale (Week 2)
**Goal:** Prove the pipeline can produce consistent quality

**Steps:**
1. Generate 50 videos with perfected settings
2. Review all 50 for consistency
3. If >90% pass: proceed to Phase 3
4. If <90% pass: return to Phase 1

**Time:** 3-5 days  
**Cost:** $25 for generation + your review time

### Phase 3: Batch Production (Weeks 3-6)
**Goal:** Generate all 9,125 videos

**Steps:**
1. Set up 24/7 automated generation
2. Generate in batches of 100 videos
3. Auto-upload to Supabase as they complete
4. Spot-check 5% for quality drift
5. Pause and adjust if quality drops

**Time:** 3-4 weeks (12-18 days of generation + review time)  
**Cost:** ~$4,500

### Phase 4: Quality Assurance (Week 7)
**Goal:** Verify all 365 days are complete and playable

**Steps:**
1. Check database: all 9,125 videos have URLs
2. Spot-test 36 lessons (10% of days)
3. Test all 5 archetypes on each spot-tested day
4. Fix any broken videos

**Time:** 3-5 days  
**Cost:** $50-100 for fixes

---

## 💰 REVISED BUDGET

### Scenario A: Keep 20% of existing videos (24 videos)
- Already spent: ~$61
- Regenerate rejected: 98 × $0.50 = $49
- Generate remaining: 9,003 × $0.50 = $4,501.50
- **Total:** ~$4,611.50

### Scenario B: Reject all existing videos
- Already spent (sunk cost): ~$61
- Generate all: 9,125 × $0.50 = $4,562.50
- **Total:** ~$4,623.50

### Scenario C: Perfect pipeline first, then generate
- Testing/iteration: $100-200
- Generate all: 9,125 × $0.50 = $4,562.50
- **Total:** ~$4,662.50-$4,762.50

**Recommendation:** Scenario C - Perfect first, then scale

---

## 🎬 WHAT TO DO RIGHT NOW

### Option 1: Create Bucket, Review Existing Videos
1. Create `kelly-videos` bucket (15 min)
2. Upload 5-10 sample videos (10 min)
3. Review them critically (30 min)
4. Decide: keep any? Or start fresh?

**Pros:** See what you have before committing  
**Cons:** Might waste 30 min if you reject all  
**Recommended if:** You want to see current quality

### Option 2: Create Bucket, Perfect Pipeline First
1. Create `kelly-videos` bucket (15 min)
2. Generate 1 test video with current settings (15 min)
3. Review and critique (10 min)
4. Iterate until perfect (2-5 days)
5. Then batch generate all 9,125 videos

**Pros:** Ensures quality before scale  
**Cons:** Delays seeing existing work  
**Recommended if:** You want perfection first

### Option 3: Create Bucket, Start Fresh
1. Create `kelly-videos` bucket (15 min)
2. Delete/ignore existing 122 videos
3. Perfect the pipeline (2-5 days)
4. Generate all 9,125 videos (3-4 weeks)

**Pros:** Clean slate, no compromises  
**Cons:** Wastes $61 already spent  
**Recommended if:** You know existing videos aren't good enough

---

## 🔧 BUCKET SETUP VERIFICATION

After creating the bucket, verify with these commands:

### Test 1: List bucket
```bash
npx supabase storage ls kelly-videos
```
**Expected:** Empty list (no error)

### Test 2: Upload test file
```bash
echo "test" > test.txt
npx supabase storage upload kelly-videos test.txt
```
**Expected:** Success message with public URL

### Test 3: Get public URL
```bash
npx supabase storage get-url kelly-videos test.txt
```
**Expected:** Public URL like `https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/kelly-videos/test.txt`

### Test 4: Access URL in browser
Open the URL from Test 3 in your browser  
**Expected:** See "test" content (not 404 or access denied)

### Test 5: Delete test file
```bash
npx supabase storage rm kelly-videos test.txt
```
**Expected:** Success message

---

## 📊 BUCKET CAPACITY PLANNING

### Storage Requirements
- **Per video:** 15-50 MB (average 30 MB for HD)
- **Total videos:** 9,125
- **Total storage:** 9,125 × 30 MB = **273.75 GB**

### Supabase Storage Limits (Free Tier)
- **Storage:** 1 GB
- **Bandwidth:** 2 GB/month

**Conclusion:** You NEED a paid Supabase plan

### Supabase Pro Plan ($25/month)
- **Storage:** 100 GB included
- **Bandwidth:** 200 GB/month
- **Additional storage:** $0.021/GB/month

**For 273.75 GB:**
- Included: 100 GB
- Additional: 173.75 GB × $0.021 = $3.65/month
- **Total:** $25 + $3.65 = **$28.65/month**

### Bandwidth Considerations
If 1,000 users watch 10 lessons each (50 videos):
- 1,000 × 50 × 30 MB = 1,500 GB bandwidth
- Pro plan: 200 GB included
- Additional: 1,300 GB × $0.09 = $117/month

**Recommendation:** Start with Pro plan, monitor bandwidth, upgrade if needed

---

## 🚀 FINAL CHECKLIST BEFORE CREATING BUCKET

- [ ] Confirmed bucket name: `kelly-videos` (not `lesson-videos`)
- [ ] Understood total video count: 9,125 (not 5,475)
- [ ] Understood total cost: ~$4,500-4,700 (not ~$2,700)
- [ ] Understood timeline: 3-4 weeks (not 2-3 weeks)
- [ ] Decided on approach: Review existing? Perfect first? Start fresh?
- [ ] Confirmed Supabase plan: Pro ($25/month minimum)
- [ ] Ready to iterate on quality until perfect
- [ ] Accepted that estimates were grossly underestimated (you were right!)

---

## 🎯 YOUR DECISION POINT

**Before I guide you through creating the bucket, tell me:**

1. **Do you want to review the 122 existing videos first?**
   - YES → Create bucket, upload samples, review, then decide
   - NO → Skip to perfecting the pipeline

2. **What's your quality bar?**
   - Kelly's face must be consistent across all videos?
   - Lipsync must be 95%+ accurate?
   - Motion must feel natural (not robotic)?
   - Sweater color must be consistent?
   - All of the above?

3. **What's your timeline priority?**
   - Launch ASAP with "good enough" → Accept some imperfection
   - Launch when PERFECT → Take 4-6 weeks to get it right
   - Launch when TIMELESS → Take as long as needed

**Your answer will determine the exact bucket setup and next steps.**

---

## 📝 NOTES FOR WHEN YOU'RE READY

When you say "create the bucket," I will:

1. Walk you through the EXACT clicks in Supabase dashboard
2. Verify the bucket is created correctly
3. Run all 5 verification tests
4. Confirm your Supabase plan can handle the storage
5. Then ask: "Upload samples? Perfect pipeline first? Or start fresh?"

**I'm ready when you are. But first: answer the 3 questions above.** 🎯

---

*This document reflects the ACTUAL requirements, not underestimates.*  
*You were right: we grossly underestimated at scale.*  
*9,125 videos. $4,500-4,700. 3-4 weeks. That's the reality.*






