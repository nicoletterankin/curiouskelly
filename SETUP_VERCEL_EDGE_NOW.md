# 🚀 Setup Vercel Edge - Copy & Paste Guide
**Follow these exact steps. Everything is copy-paste ready.**

**⚠️ AGENT DIRECTIVE:** This task is assigned to **Infrastructure Agent**. Follow `BOSS_OPERATIONAL_MANUAL.md` for operational rules. Report completion and any blockers.

---

## ✅ STEP 1: Open Vercel Dashboard

**Click this link:**
```
https://vercel.com/dashboard
```

**Or copy-paste:** `https://vercel.com/dashboard`

---

## ✅ STEP 2: Select Your Project

1. In the dashboard, find and click: **`curiouskelly`**
2. You should see project settings

---

## ✅ STEP 3: Create Edge Config

### 3.1 Go to Storage
1. In the left sidebar, click: **`Storage`**
2. You'll see tabs: `Blob`, `Edge Config`, `KV`, etc.

### 3.2 Create Edge Config
1. Click the **`Edge Config`** tab
2. Click the button: **`Create Edge Config`**
3. In the name field, type exactly:
   ```
   curious-kelly-lessons
   ```
4. Click: **`Create`**

### 3.3 Copy Connection String
1. After creation, you'll see a connection string that looks like:
   ```
   https://edge-config.vercel.app/ecfg_xxxxxxxxxxxxx
   ```
2. **COPY THIS ENTIRE STRING** (you'll need it in Step 5)

---

## ✅ STEP 4: Create Blob Storage Buckets

### 4.1 Go to Blob Storage
1. Click the **`Blob`** tab (in Storage section)
2. Click: **`Create Bucket`**

### 4.2 Create First Bucket
1. In the name field, type exactly:
   ```
   curious-kelly-videos
   ```
2. Toggle **`Public`** to **ON** (or check the Public checkbox)
3. Click: **`Create`**

### 4.3 Create Second Bucket
1. Click: **`Create Bucket`** again
2. Name:
   ```
   curious-kelly-audio
   ```
3. Toggle **`Public`** to **ON**
4. Click: **`Create`**

### 4.4 Create Third Bucket
1. Click: **`Create Bucket`** again
2. Name:
   ```
   curious-kelly-visuals
   ```
3. Toggle **`Public`** to **ON**
4. Click: **`Create`**

**You should now have 3 buckets total.**

---

## ✅ STEP 5: Set Environment Variables

### 5.1 Go to Environment Variables
1. In the left sidebar, click: **`Settings`**
2. Click: **`Environment Variables`**

### 5.2 Add EDGE_CONFIG Variable

**Copy-paste these exact values:**

**Key:**
```
EDGE_CONFIG
```

**Value:**
```
PASTE_YOUR_CONNECTION_STRING_HERE
```
*(Replace `PASTE_YOUR_CONNECTION_STRING_HERE` with the connection string you copied in Step 3.3)*

**Environments:**
- ✅ Check: **Production**
- ✅ Check: **Preview**
- ✅ Check: **Development**

Click: **`Save`**

### 5.3 Add EDGE_CONFIG_SYNC_SECRET Variable

**Generate Secret (PowerShell):**

Open PowerShell and run:
```powershell
$bytes = New-Object byte[] 32
[System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
[Convert]::ToHexString($bytes).ToLower()
```

**Copy the output** (it will be a long string like: `a1b2c3d4e5f6...`)

**Back in Vercel Dashboard:**

**Key:**
```
EDGE_CONFIG_SYNC_SECRET
```

**Value:**
```
PASTE_YOUR_GENERATED_SECRET_HERE
```
*(Replace with the secret you just generated)*

**Environments:**
- ✅ Check: **Production**
- ✅ Check: **Preview**
- ✅ Check: **Development**

Click: **`Save`**

---

## ✅ STEP 6: Verify Setup

### 6.1 Check Environment Variables

**In PowerShell, run:**
```powershell
vercel env ls
```

**You should see:**
- `EDGE_CONFIG`
- `EDGE_CONFIG_SYNC_SECRET`

### 6.2 Test Edge Config Sync

**Run this command:**
```powershell
$env:EDGE_CONFIG_SYNC_SECRET = "YOUR_SECRET_HERE"
curl -X POST https://curiouskelly.com/api/sync-edge-config -H "Content-Type: application/json" -d "{\"secret\":\"$env:EDGE_CONFIG_SYNC_SECRET\"}"
```

*(Replace `YOUR_SECRET_HERE` with your actual secret)*

**Expected response:**
```json
{"success":true,"synced":365,"message":"Synced 365 lessons to Edge Config"}
```

---

## ✅ STEP 7: Migrate Assets (Optional - Later)

**Preview what will be migrated:**
```powershell
npx tsx scripts/migrate-to-blob.ts --dry-run
```

**Migrate all assets:**
```powershell
npx tsx scripts/migrate-to-blob.ts --all
```

---

## ✅ CHECKLIST

Before moving on, verify:

- [ ] Edge Config created: `curious-kelly-lessons`
- [ ] Connection string copied
- [ ] Blob bucket created: `curious-kelly-videos`
- [ ] Blob bucket created: `curious-kelly-audio`
- [ ] Blob bucket created: `curious-kelly-visuals`
- [ ] Environment variable set: `EDGE_CONFIG`
- [ ] Environment variable set: `EDGE_CONFIG_SYNC_SECRET`
- [ ] Sync test successful (Step 6.2)

---

## 🎯 QUICK REFERENCE

**Dashboard Links:**
- Main Dashboard: `https://vercel.com/dashboard`
- Project Settings: `https://vercel.com/dashboard/lotd/curiouskelly/settings`
- Storage: `https://vercel.com/dashboard/lotd/curiouskelly/storage`
- Environment Variables: `https://vercel.com/dashboard/lotd/curiouskelly/settings/environment-variables`

**Commands:**
```powershell
# Check environment variables
vercel env ls

# Sync Edge Config (after setup)
curl -X POST https://curiouskelly.com/api/sync-edge-config -H "Content-Type: application/json" -d "{\"secret\":\"YOUR_SECRET\"}"

# Preview migration
npx tsx scripts/migrate-to-blob.ts --dry-run
```

---

## ❓ TROUBLESHOOTING

**Problem:** Can't find "Storage" in sidebar
- **Solution:** Make sure you're in the project settings, not team settings

**Problem:** Edge Config creation fails
- **Solution:** Make sure you're on a paid Vercel plan (Edge Config requires Pro plan)

**Problem:** Environment variables not showing up
- **Solution:** Make sure you selected all environments (Production, Preview, Development)

**Problem:** Sync fails with "Unauthorized"
- **Solution:** Check that `EDGE_CONFIG_SYNC_SECRET` matches what you set in environment variables

---

**That's it! You're done! 🎉**

---## 📋 COMPLETION REPORT (Infrastructure Agent)**After completing all steps, fill this out:**

- [ ] All checklist items completed
- [ ] Sync test successful (365 lessons synced)
- [ ] No errors or blockers encountered
- [ ] All environment variables verified
- [ ] Ready for Boss approval

**Report Format:**
```
✅ VERCEL EDGE SETUP COMPLETEAgent: Infrastructure Agent
Completed: [Date/Time]
Status: SUCCESS / PARTIAL / BLOCKEDCompleted Steps:
- [ ] Step 1: Dashboard access
- [ ] Step 2: Edge Config created
- [ ] Step 3: Blob buckets created (3/3)
- [ ] Step 4: Environment variables set (2/2)
- [ ] Step 5: Verification successfulIssues/Blockers:
- [List any issues or deviations]Next Steps:
- [What should happen next?]Ready for: [Migration / Testing / Boss Review]
```---