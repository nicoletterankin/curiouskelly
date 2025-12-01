# 🚀 Kelly Visual Identity - QUICK START

**Get from zero to production in 24 hours**

---

## ⏱️ TODAY (1-2 Hours)

### Step 1: Install Dependencies (5 minutes)
```powershell
cd C:\Users\user\UI-TARS-desktop
.\scripts\kelly-visual-identity\install-dependencies.ps1
```

### Step 2: Set Up Environment (5 minutes)
```powershell
# Copy template
cp scripts/kelly-visual-identity/env-template.txt .env.local

# Edit .env.local and fill in:
# - CLOUDFLARE_R2_ACCESS_KEY_ID (get from Step 3)
# - CLOUDFLARE_R2_SECRET_ACCESS_KEY (get from Step 3)
# - GOOGLE_AI_API_KEY (already provided: AIzaSyBVPxRvxDfA07qyAjbZ6FfRqo5L_rxquHE)
# - Supabase credentials (already in your existing .env)
```

### Step 3: Create R2 Bucket (10 minutes)
1. Go to https://dash.cloudflare.com/
2. Navigate to **R2 Object Storage**
3. Click **Create bucket** → Name: `kelly-assets` → Create
4. Click **Manage R2 API Tokens** → **Create API token**
5. Name: `kelly-assets-access`
6. Permissions: **Object Read & Write**
7. Scope: `kelly-assets` bucket only
8. **COPY THE KEYS IMMEDIATELY** → Add to `.env.local`

### Step 4: Deploy Database (5 minutes)
```powershell
# Option A: Using Supabase CLI
supabase db push supabase/migrations/20251130_create_kelly_assets.sql

# Option B: Manual
# 1. Go to https://supabase.com/dashboard
# 2. Open SQL Editor
# 3. Copy/paste contents of: supabase/migrations/20251130_create_kelly_assets.sql
# 4. Run
```

### Step 5: Prepare LoRA Dataset (10 minutes)
```powershell
npx tsx scripts/kelly-visual-identity/prepare-lora-dataset.ts
```
**Output:** `lora-training-dataset/` folder with 7 images + captions

### Step 6: Start LoRA Training (15 minutes setup, 4-6 hours training)
1. Go to https://civitai.com/models/train
2. Log in: `nicoletterankin201` (Pro account)
3. Click **New Training**
4. Upload entire `lora-training-dataset/` folder
5. Configure:
   - Base model: **FLUX.1 Dev**
   - Instance prompt: `kelly`
   - Class prompt: `woman`
   - Training steps: **1500-2000**
   - Learning rate: **1e-4**
6. Click **Start Training** (Cost: ~$15-25)
7. Training runs overnight (4-6 hours)

### Step 7: Test Generation (Optional - 20 minutes)
```powershell
# Generate 3 test poses to verify pipeline works
npx tsx scripts/kelly-visual-identity/generate-kelly-poses.ts
```
**Note:** Without LoRA, consistency will vary. Full generation happens tomorrow.

---

## 🌅 TOMORROW (2-3 Hours)

### Step 8: Download Trained LoRA
1. Check Civitai training status
2. Download: `kelly_lora_v1.safetensors`
3. Save to: `models/lora/kelly_lora_v1.safetensors`

### Step 9: Generate All 12 Poses
```powershell
# Update script to use LoRA, then generate
npx tsx scripts/kelly-visual-identity/generate-kelly-poses.ts --use-lora
```
**Target:** 5 variations per pose = 60 images

### Step 10: Review & Select Best
1. Open `generated-poses/` folder
2. Review all 60 images
3. Select best version of each pose
4. Rename with `_hero` suffix

### Step 11: Upload to Production
```powershell
npx tsx scripts/kelly-visual-identity/upload-to-r2.ts generated-poses/
```

### Step 12: Mark as Published
```sql
-- In Supabase SQL Editor
UPDATE kelly_assets 
SET status = 'published', is_hero = true, published_at = NOW()
WHERE filename LIKE '%_hero.png';
```

### Step 13: Integrate Component
```tsx
// In your lesson player
import KellyAvatar from "@/components/KellyAvatar";

<KellyAvatar state="thinking" layout="horizontal" priority={true} />
```

### Step 14: Test
- Desktop (1920x1080)
- Tablet (768x1024)
- Mobile (375x667)

---

## 📚 Full Documentation

- **Complete Guide:** `scripts/kelly-visual-identity/README.md`
- **Execution Checklist:** `KELLY_VISUAL_IDENTITY_EXECUTION_CHECKLIST.md`
- **Summary:** `KELLY_VISUAL_IDENTITY_COMPLETE.md`
- **R2 Setup:** `scripts/kelly-visual-identity/r2-setup-guide.md`
- **Examples:** `examples/kelly-avatar-usage.tsx`

---

## 🆘 Need Help?

**Owner:** Nicolette (nicoletterankin@gmail.com)  
**Support:** hello@curiouskelly.com

**Common Issues:**
- Dependencies not installing? Run as Administrator
- R2 access denied? Check API token permissions
- Supabase error? Verify credentials in `.env.local`
- Generation failing? Check Google AI API key

---

## ✅ Checklist

**TODAY:**
- [ ] Dependencies installed
- [ ] Environment configured
- [ ] R2 bucket created
- [ ] Database deployed
- [ ] LoRA dataset prepared
- [ ] Civitai training started

**TOMORROW:**
- [ ] LoRA downloaded
- [ ] All 12 poses generated
- [ ] Best versions selected
- [ ] Uploaded to production
- [ ] Component integrated
- [ ] Responsive layout tested

---

**🎯 Result:** Production-ready Kelly avatar system in 24 hours!



