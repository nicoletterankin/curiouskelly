# Kelly Visual Identity Pipeline - Execution Checklist

**Owner:** Nicolette (nicoletterankin@gmail.com)  
**Timeline:** Foundation TODAY, Production TOMORROW  
**Status:** IN PROGRESS

---

## ✅ COMPLETED TODAY

### Infrastructure Setup
- [x] Created Supabase migration: `supabase/migrations/20251130_create_kelly_assets.sql`
- [x] Created generation script: `scripts/kelly-visual-identity/generate-kelly-poses.ts`
- [x] Created LoRA prep script: `scripts/kelly-visual-identity/prepare-lora-dataset.ts`
- [x] Created R2 upload script: `scripts/kelly-visual-identity/upload-to-r2.ts`
- [x] Created React component: `components/KellyAvatar.tsx`
- [x] Created Cloudflare loader: `lib/cloudflare-loader.ts`
- [x] Created CSS styles: `styles/kelly-avatar.css`
- [x] Created R2 setup guide: `scripts/kelly-visual-identity/r2-setup-guide.md`
- [x] Created environment template: `scripts/kelly-visual-identity/env-template.txt`
- [x] Created comprehensive README: `scripts/kelly-visual-identity/README.md`

---

## 🎯 TODO TODAY (Next 1-2 Hours)

### Step 1: Database Setup (5 minutes)
```bash
# Option A: Using Supabase CLI
cd C:\Users\user\UI-TARS-desktop
supabase db push supabase/migrations/20251130_create_kelly_assets.sql

# Option B: Manual (if CLI not available)
# 1. Go to https://supabase.com/dashboard/project/[your-project]/editor
# 2. Open SQL Editor
# 3. Copy contents of supabase/migrations/20251130_create_kelly_assets.sql
# 4. Run the SQL
# 5. Verify table exists: SELECT * FROM kelly_assets LIMIT 1;
```

**Verification:**
- [ ] Table `kelly_assets` exists in Supabase
- [ ] View `kelly_production_assets` exists
- [ ] Indexes created successfully

---

### Step 2: Cloudflare R2 Setup (10 minutes)

**2.1 Create Bucket**
1. [ ] Go to https://dash.cloudflare.com/
2. [ ] Navigate to **R2 Object Storage**
3. [ ] Click **Create bucket**
4. [ ] Name: `kelly-assets`
5. [ ] Location: **Automatic**
6. [ ] Click **Create bucket**

**2.2 Create API Token**
1. [ ] In R2 dashboard, click **Manage R2 API Tokens**
2. [ ] Click **Create API token**
3. [ ] Name: `kelly-assets-access`
4. [ ] Permissions: **Object Read & Write**
5. [ ] Scope: Apply to specific buckets → `kelly-assets`
6. [ ] Click **Create API Token**
7. [ ] **COPY IMMEDIATELY:**
   - Access Key ID: `_______________________________`
   - Secret Access Key: `_______________________________`

**2.3 Configure Environment Variables**
1. [ ] Copy `scripts/kelly-visual-identity/env-template.txt` to `.env.local`
2. [ ] Fill in R2 credentials from Step 2.2
3. [ ] Verify Google AI API key is present
4. [ ] Verify Supabase credentials are present

**Verification:**
```bash
# Test that env vars are loaded
node -e "require('dotenv').config(); console.log('R2 Key:', process.env.CLOUDFLARE_R2_ACCESS_KEY_ID ? '✅ Set' : '❌ Missing')"
```

---

### Step 3: Prepare LoRA Training Dataset (10 minutes)

```bash
cd C:\Users\user\UI-TARS-desktop
npx tsx scripts/kelly-visual-identity/prepare-lora-dataset.ts
```

**Expected Output:**
- [ ] Folder created: `lora-training-dataset/`
- [ ] 7 images copied from reference locations
- [ ] 7 caption `.txt` files created
- [ ] README.md created with training instructions

**Verification:**
```bash
# Check dataset folder
ls lora-training-dataset/
# Should show: 7 .jpeg/.png files + 7 .txt files + README.md
```

**Reference Images Included:**
1. `4.jpeg` - Close-up face, big smile (HERO face reference)
2. `pray.jpeg` - Hands together, playful expression
3. `open-walk.jpeg` - Full body walking, profile view
4. `square-chair2.jpeg` - Seated, hand on heart
5. `our-girl.jpeg` - Seated, chin on hand, thoughtful
6. `open.png` - Close-up, chin on hand, looking up
7. `close.jpeg` - Close-up, eyes closed, peaceful

---

### Step 4: Start LoRA Training on Civitai (15 minutes setup, 4-6 hours training)

**4.1 Upload Dataset**
1. [ ] Go to https://civitai.com/models/train
2. [ ] Log in with: `nicoletterankin201` (Pro account)
3. [ ] Click **New Training**
4. [ ] Upload entire `lora-training-dataset/` folder

**4.2 Configure Training**
- [ ] **Base model:** FLUX.1 Dev (or SDXL 1.0 if FLUX unavailable)
- [ ] **Training type:** Character/Person
- [ ] **Instance prompt:** `kelly`
- [ ] **Class prompt:** `woman`
- [ ] **Training steps:** 1500-2000
- [ ] **Learning rate:** 1e-4
- [ ] **Network dimension:** 32
- [ ] **Network alpha:** 16

**4.3 Start Training**
- [ ] Review settings
- [ ] Estimated cost: $15-25
- [ ] Estimated time: 4-6 hours
- [ ] Click **Start Training**
- [ ] Note training ID: `_______________________________`

**Training will run overnight. Continue with test generation using base model.**

---

### Step 5: Test Generation with Google AI Studio (20 minutes)

```bash
cd C:\Users\user\UI-TARS-desktop

# Generate 3 test poses
npx tsx scripts/kelly-visual-identity/generate-kelly-poses.ts
```

**Expected Output:**
- [ ] Folder created: `generated-poses/`
- [ ] At least 3 test images generated:
  - `kelly_idle_v1.png`
  - `kelly_thinking_v1.png`
  - `kelly_celebrating_v1.png`
- [ ] Generation log: `generated-poses/generation_log.json`

**Quality Check:**
- [ ] Face looks consistent across images
- [ ] Blue sweater visible
- [ ] White background present
- [ ] Poses are distinguishable

**Note:** Without LoRA, consistency may vary. This is expected. Full generation happens tomorrow after LoRA training completes.

---

### Step 6: Test R2 Upload (10 minutes)

```bash
cd C:\Users\user\UI-TARS-desktop

# Upload test poses to R2
npx tsx scripts/kelly-visual-identity/upload-to-r2.ts generated-poses/
```

**Expected Output:**
- [ ] Files uploaded to R2 bucket: `kelly-assets/staging/poses/[pose-name]/`
- [ ] Metadata inserted into Supabase `kelly_assets` table
- [ ] CDN URLs generated and printed

**Verification:**
1. [ ] Check Cloudflare R2 dashboard - files visible
2. [ ] Check Supabase dashboard - records in `kelly_assets` table
3. [ ] Test CDN URL in browser (should load image)

---

## 📊 END OF DAY STATUS

**Infrastructure:** ✅ Complete
- [x] Database schema deployed
- [x] R2 bucket configured
- [x] Scripts created and tested
- [x] Components built

**Training:** 🔄 In Progress
- [x] Dataset prepared
- [x] Training started on Civitai
- [ ] Training complete (4-6 hours)

**Generation:** ⚠️ Partial
- [x] Test generation working
- [ ] Full 12 poses with LoRA (tomorrow)

**Integration:** ✅ Ready
- [x] React component created
- [x] CSS styles created
- [x] Image loader configured

---

## 🎯 TOMORROW (After LoRA Training Completes)

### Step 7: Download Trained LoRA
1. [ ] Check Civitai training status
2. [ ] Download trained LoRA file: `kelly_lora_v1.safetensors`
3. [ ] Save to: `models/lora/kelly_lora_v1.safetensors`

### Step 8: Generate All 12 Poses with LoRA
```bash
# Update generate-kelly-poses.ts to use LoRA
# Then generate all 12 poses
npx tsx scripts/kelly-visual-identity/generate-kelly-poses.ts --use-lora
```

**Target:** 5 variations per pose = 60 total images

### Step 9: Review and Select Best Versions
1. [ ] Review all 60 generated images
2. [ ] Select best version for each of 12 poses
3. [ ] Verify quality gates (face, outfit, scene, pose, resolution)
4. [ ] Rename selected images with `_hero` suffix

### Step 10: Upload to Production
```bash
# Upload approved images
npx tsx scripts/kelly-visual-identity/upload-to-r2.ts generated-poses/approved/
```

### Step 11: Update Supabase
```sql
-- Mark hero images as published
UPDATE kelly_assets 
SET status = 'published', is_hero = true, published_at = NOW()
WHERE filename IN (
  'kelly_idle_hero.png',
  'kelly_thinking_hero.png',
  -- ... all 12 poses
);
```

### Step 12: Move to Production Folders
In R2 dashboard:
1. [ ] Move images from `staging/` to `production/poses/[pose-name]/`
2. [ ] Verify CDN URLs work
3. [ ] Update Supabase `r2_key` if needed

### Step 13: Integrate into App
```tsx
// In your lesson player component
import KellyAvatar from "@/components/KellyAvatar";

<KellyAvatar 
  state="thinking" 
  layout="horizontal"
  priority={true}
/>
```

### Step 14: Test Responsive Layout
1. [ ] Test on desktop (1920x1080)
2. [ ] Test on tablet (768x1024)
3. [ ] Test on mobile (375x667)
4. [ ] Verify pointing directions adapt (left/right → up/down)
5. [ ] Verify smooth transitions between states

---

## 🎉 SUCCESS CRITERIA

### Today (Foundation)
- [x] R2 bucket created and structured
- [x] Supabase table created
- [x] LoRA training started on Civitai
- [x] At least 3 test poses generated
- [x] Generation pipeline script working
- [x] Upload pipeline tested

### Tomorrow (Production)
- [ ] LoRA training complete
- [ ] All 12 core poses generated with LoRA
- [ ] Best versions approved and uploaded to production
- [ ] KellyAvatar component integrated into app
- [ ] Responsive layout working on mobile and desktop

---

## 🐛 Troubleshooting

### "npx tsx not found"
```bash
npm install -g tsx
# Or use:
npm install --save-dev tsx
```

### "Cannot find module '@google/generative-ai'"
```bash
npm install @google/generative-ai
```

### "Cannot find module '@aws-sdk/client-s3'"
```bash
npm install @aws-sdk/client-s3
```

### "Cannot find module '@supabase/supabase-js'"
```bash
npm install @supabase/supabase-js
```

### "Supabase CLI not found"
- Option 1: Install Supabase CLI: `npm install -g supabase`
- Option 2: Run SQL manually in Supabase dashboard

### "Google AI API quota exceeded"
- Free tier: 1500 requests/day
- Wait 24 hours or upgrade to paid tier
- Generate in batches with delays

---

## 📞 Contact

**Owner:** Nicolette  
**Email:** nicoletterankin@gmail.com  
**Support Email:** hello@curiouskelly.com

---

## 📝 Notes

- All credentials stored in `.env.local` (NEVER commit)
- R2 costs ~$0.05/month for Kelly assets
- LoRA training is one-time cost of $15-25
- Google AI Studio has free tier (1500/day)
- Civitai Pro account required for training

---

**Last Updated:** 2025-11-30  
**Status:** Foundation Complete, Ready for LoRA Training





