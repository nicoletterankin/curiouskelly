# Navigation to Quotas Filter - CORRECT PATH

**IMPORTANT:** The filter you need is ON THE QUOTAS PAGE, not the general search!

---

## 🎯 CORRECT NAVIGATION PATH

### Step 1: Go to Quotas Page
**Direct Link:**
```
https://console.cloud.google.com/apis/api/aiplatform.googleapis.com/quotas?project=gen-lang-client-0005524332
```

**Or navigate:**
1. APIs & Services → Enabled APIs & services
2. Click "Vertex AI API"
3. Click "Quotas & System Limits" tab

---

### Step 2: Find the Filter Box IN THE TABLE

**NOT the search bar at the top of the page!**

Look for:
- **Filter box** that says "Filter Enter property name or value"
- **Location:** Above the quotas TABLE (the big table with 12,039 quotas)
- **Inside:** The quotas table area, not the page header

---

### Step 3: Filter the Quotas Table

1. **Click the filter box** (inside the quotas table area)
2. **Type:** `imagen`
3. **Press Enter**

This will filter the 12,039 quotas to show only Imagen-related ones.

---

## ❌ COMMON MISTAKE

**Don't use:**
- ❌ The main search bar at the top (shows documentation)
- ❌ General Cloud Console search

**Do use:**
- ✅ The filter box INSIDE the quotas table
- ✅ On the "Quotas & System Limits" tab

---

## 🔍 WHAT YOU'RE LOOKING FOR

After filtering, you should see quotas like:
- "Online prediction requests per base model per minute per region"
- With dimension: `region: us-central1`
- With base model: `imagen-3.0-generate-002`
- Current value: 10-20 (your limit)

---

**Status:** Need to navigate back to quotas page and use table filter











