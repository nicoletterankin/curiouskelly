# Vercel Dashboard Configuration Guide
**Automated Setup via API Script**

---

## 🚀 Quick Setup (Automated)

### Option 1: Use API Script (Recommended)

1. **Get Vercel Token:**
   - Go to: https://vercel.com/account/tokens
   - Click "Create Token"
   - Name: `Edge Optimization Setup`
   - Copy the token

2. **Set Environment Variable:**
   ```powershell
   $env:VERCEL_TOKEN = "your-token-here"
   ```

3. **Run Configuration Script:**
   ```powershell
   npm run configure:vercel-edge
   ```

This script will:
- ✅ Create Edge Config (`curious-kelly-lessons`)
- ✅ Create Blob Storage buckets (videos, audio, visuals)
- ✅ Set environment variables automatically

---

## 🎯 Manual Setup (If Script Fails)

### Step 1: Create Edge Config

1. Go to: https://vercel.com/dashboard
2. Select project: **curiouskelly**
3. Go to **Storage** → **Edge Config**
4. Click **"Create Edge Config"**
5. Name: `curious-kelly-lessons`
6. Click **"Create"**
7. **Copy the Connection String** (looks like: `https://edge-config.vercel.app/...`)

### Step 2: Create Blob Storage Buckets

1. In same project, go to **Storage** → **Blob**
2. Click **"Create Bucket"**
3. Create three buckets:
   - `curious-kelly-videos` (Public: Yes)
   - `curious-kelly-audio` (Public: Yes)
   - `curious-kelly-visuals` (Public: Yes)

### Step 3: Set Environment Variables

1. Go to **Settings** → **Environment Variables**
2. Add these variables for **Production**, **Preview**, and **Development**:

**EDGE_CONFIG:**
- Key: `EDGE_CONFIG`
- Value: `<connection-string-from-step-1>`
- Environments: Production, Preview, Development

**EDGE_CONFIG_SYNC_SECRET:**
- Key: `EDGE_CONFIG_SYNC_SECRET`
- Value: `<generate-random-secret>` (use script below)
- Environments: Production, Preview, Development

**Generate Secret (PowerShell):**
```powershell
$bytes = New-Object byte[] 32
[System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
[Convert]::ToHexString($bytes).ToLower()
```

---

## ✅ Verification

After setup, verify configuration:

```powershell
# Check environment variables
vercel env ls

# Test Edge Config sync
npm run sync-edge-config
```

---

## 🎯 Next Steps

After configuration:

1. **Run Initial Sync:**
   ```powershell
   npm run sync-edge-config
   ```

2. **Migrate Assets:**
   ```powershell
   npx tsx scripts/migrate-to-blob.ts --dry-run  # Preview
   npx tsx scripts/migrate-to-blob.ts --all      # Migrate all
   ```

3. **Test Performance:**
   - Visit: https://curiouskelly.com/api/lessons/1-edge
   - Should return data with `_source: "edge-config"` or `_source: "supabase"`

---

**Status:** Ready for configuration  
**Script:** `scripts/configure-vercel-edge.ts`  
**Manual Guide:** Follow steps above if script fails

