# ✅ KELLY VISUAL IDENTITY - SIMPLE INSTRUCTIONS

I've done everything I can automatically. Here's what's left for you to do:

---

## ✅ DONE AUTOMATICALLY

- ✅ Created `lora-training-dataset` folder with 7 images + 7 caption files
- ✅ Created database schema SQL file
- ✅ Created all React components and scripts
- ✅ Created documentation

---

## 🎯 YOUR MANUAL STEPS (3 THINGS)

### STEP 1: Set Up Cloudflare R2 Bucket (10 minutes)

1. **Go to:** https://dash.cloudflare.com/47ebb2a1adc311cb106acc89720e352c
2. **Click:** "R2 Object Storage" in left sidebar
3. **Click:** "Create bucket" button
4. **Name:** `kelly-assets`
5. **Click:** "Create bucket"

**Then create API token:**

6. **Click:** "Manage R2 API Tokens"
7. **Click:** "Create API token"
8. **Name:** `kelly-assets-access`
9. **Permissions:** Check "Object Read & Write"
10. **Scope:** Select "kelly-assets" bucket only
11. **Click:** "Create API Token"
12. **COPY BOTH KEYS** (you'll need them - save in a text file temporarily)

---

### STEP 2: Set Up Supabase Database (5 minutes)

1. **Go to:** https://supabase.com/dashboard
2. **Click:** Your project
3. **Click:** "SQL Editor" in left sidebar
4. **Click:** "New query"
5. **Copy this SQL:**

```sql
-- Kelly Visual Identity Asset Management System
CREATE TABLE IF NOT EXISTS kelly_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255) NOT NULL,
    r2_key TEXT NOT NULL,
    r2_bucket VARCHAR(100) DEFAULT 'kelly-assets',
    pose_type VARCHAR(50) NOT NULL,
    pose_direction VARCHAR(20),
    emotion VARCHAR(50),
    status VARCHAR(20) DEFAULT 'draft' CHECK (status IN ('draft', 'review', 'approved', 'published', 'archived')),
    is_hero BOOLEAN DEFAULT false,
    version INTEGER DEFAULT 1,
    generation_model VARCHAR(100),
    generation_prompt TEXT,
    generation_seed VARCHAR(100),
    generation_params JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    approved_at TIMESTAMPTZ,
    published_at TIMESTAMPTZ,
    CONSTRAINT unique_hero_per_pose UNIQUE (pose_type, pose_direction) WHERE is_hero = true AND status = 'published'
);

CREATE INDEX IF NOT EXISTS idx_kelly_assets_pose ON kelly_assets(pose_type, pose_direction);
CREATE INDEX IF NOT EXISTS idx_kelly_assets_status ON kelly_assets(status);
CREATE INDEX IF NOT EXISTS idx_kelly_assets_hero ON kelly_assets(is_hero) WHERE is_hero = true;

CREATE OR REPLACE VIEW kelly_production_assets AS
SELECT 
    id, pose_type, pose_direction, emotion, filename,
    CONCAT('https://kelly-assets.curiouskelly.com/', r2_key) as cdn_url,
    r2_key, created_at, published_at
FROM kelly_assets
WHERE status = 'published' AND is_hero = true;
```

6. **Paste** the SQL into the editor
7. **Click:** "Run" (or press F5)
8. **Verify:** You see "Success. No rows returned"

---

### STEP 3: Start Civitai LoRA Training (15 minutes)

1. **Go to:** https://civitai.com/models/train
2. **Log in with:** nicoletterankin201
3. **Click:** "New Training"
4. **Open folder:** `C:\Users\user\UI-TARS-desktop\lora-training-dataset`
5. **Drag ALL 15 files** (7 images + 7 .txt files + README) into Civitai upload area

**Configure training:**

6. **Base model:** Select "FLUX.1 Dev" (or "SDXL 1.0" if FLUX unavailable)
7. **Training type:** Character/Person
8. **Instance prompt:** `kelly`
9. **Class prompt:** `woman`
10. **Training steps:** `1500` to `2000`
11. **Learning rate:** `1e-4` (or `0.0001`)
12. **Network dimension:** `32`
13. **Network alpha:** `16`
14. **Click:** "Start Training"

**Cost:** ~$15-25  
**Time:** 4-6 hours (you'll get an email when done)

---

## 🌅 AFTER CIVITAI EMAIL (Tomorrow)

When you get the email that training is complete:

### STEP 4: Generate Kelly Poses

1. **Download** the trained LoRA from Civitai (save as `kelly_lora_v1.safetensors`)
2. **Open PowerShell** in `C:\Users\user\UI-TARS-desktop`
3. **Run:**
   ```powershell
   npx tsx scripts/kelly-visual-identity/generate-kelly-poses.ts
   ```

### STEP 5: Upload to R2

```powershell
npx tsx scripts/kelly-visual-identity/upload-to-r2.ts generated-poses/
```

### STEP 6: Mark as Published

In Supabase SQL Editor, run:

```sql
UPDATE kelly_assets 
SET status = 'published', is_hero = true, published_at = NOW()
WHERE status = 'review';
```

### STEP 7: Use in Your App

```tsx
import KellyAvatar from "@/components/KellyAvatar";

<KellyAvatar state="thinking" layout="horizontal" priority={true} />
```

---

## 📁 WHAT'S IN THE FOLDERS

- **`lora-training-dataset/`** - 7 Kelly reference images + captions (ready to upload to Civitai)
- **`components/KellyAvatar.tsx`** - React component (ready to use)
- **`lib/cloudflare-loader.ts`** - Image optimization
- **`styles/kelly-avatar.css`** - Responsive styles
- **`examples/kelly-avatar-usage.tsx`** - 7 integration examples

---

## 🆘 NEED HELP?

- **Full docs:** `KELLY_VISUAL_IDENTITY_COMPLETE.md`
- **Quick start:** `KELLY_QUICK_START.md`
- **Email:** hello@curiouskelly.com

---

## ✅ CHECKLIST

**TODAY:**
- [ ] Step 1: Create R2 bucket and get API keys
- [ ] Step 2: Run SQL in Supabase
- [ ] Step 3: Upload to Civitai and start training

**TOMORROW (after email):**
- [ ] Step 4: Generate poses
- [ ] Step 5: Upload to R2
- [ ] Step 6: Mark as published
- [ ] Step 7: Integrate component

---

**That's it! 3 steps today, 4 steps tomorrow. Total time: ~1 hour.**

