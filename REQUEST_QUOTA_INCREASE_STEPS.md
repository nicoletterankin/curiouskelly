# Request Quota Increase - Final Steps

**Status:** ✅ Filter applied - Can see imagen quotas  
**Next:** Find specific quota and request increase

---

## 🎯 FIND YOUR SPECIFIC QUOTA

You can see multiple imagen quotas. You need to find:

**Look for:**
- **Name:** `base_model:imagen-3.0-generate`
- **Dimensions:** `region: us-central1`
- **Value:** `100` (current limit)
- **Adjustable:** `Yes`

**Note:** You might need to scroll down in the table to find `us-central1` region. The quotas are listed for many regions (africa-south1, asia-east1, etc.).

---

## ✏️ REQUEST INCREASE

### Step 1: Click Actions Menu
- Find the row for `base_model:imagen-3.0-generate` with `region: us-central1`
- Click the **three dots (⋮)** in the "Actions" column

### Step 2: Select Edit Quota
- Choose **"Edit Quota"** or **"Request Increase"** from the menu

### Step 3: Enter New Limit
**Recommended values:**
- **Development/Testing:** `200` requests/minute
- **Production:** `500` requests/minute
- **High-volume batch:** `1000` requests/minute

### Step 4: Add Justification
**Copy this:**
```
Generating game assets for production use. Batch generation 
of character assets requires higher throughput. Reference 
image-based generation increases API usage per request. 
Need quota increase to support production asset pipeline.
```

### Step 5: Submit
- Click **"Submit"** or **"Request Increase"**

---

## ⏱️ WHAT TO EXPECT

1. **Auto-approval:** Small increases (< 2x) may be instant
2. **Manual review:** Larger increases take 24-48 hours
3. **Email notification:** You'll receive email when approved

---

## ✅ AFTER APPROVAL

Once approved:
1. **Test reference images** with `test_reference_fix.ps1`
2. **Regenerate Kelly assets** with working reference images
3. **Generate all missing assets** for Reinmaker

---

**Status:** Ready to request increase  
**Priority:** HIGH - Blocking reference image testing











