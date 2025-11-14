# Step-by-Step: Request Imagen 3.0 Quota Increase

**Current Page:** ✅ Quotas & System Limits - Vertex AI API  
**Next Step:** Find the specific Imagen quota

---

## 🔍 STEP 1: Filter for Imagen Quota

**In the filter bar at the top of the quotas table:**

1. **Click the filter box** (says "Filter Enter property name or value")

2. **Type one of these:**
   - `imagen`
   - `online_prediction_requests_per_base_model`
   - `prediction_requests`

3. **Press Enter** or click outside the filter box

**This will narrow down the 12,039 quotas to just the ones related to Imagen/prediction requests.**

---

## 🎯 STEP 2: Find the Right Quota

Look for a quota with:
- **Name:** Contains "online prediction requests per base model" or "imagen"
- **Dimensions:** Should show `region: us-central1` (or your location)
- **Base model:** Should mention `imagen-3.0-generate` or similar

**Example name might be:**
- "Online prediction requests per base model per minute per region"
- "Imagen generate requests per minute"
- "Prediction requests per base model"

---

## ✏️ STEP 3: Request Increase

Once you find the quota:

1. **Click the three dots (⋮)** in the "Action" column for that quota row
2. **Select "Edit Quota"** or "Request Increase"
3. **Enter new limit:**
   - **Recommended:** `100` or `200` requests/minute
   - **For production:** `500` or `1000` requests/minute
4. **Fill in justification:**
   ```
   Generating game assets for production use. Batch generation 
   of character assets requires higher throughput. Reference 
   image-based generation increases API usage per request.
   ```
5. **Submit request**

---

## 🔍 ALTERNATIVE: Search by Metric Name

If the filter doesn't work, try:

1. **Click "View quotas & system limits"** link (below the summary stats)
2. **Use the filter with:** `aiplatform.googleapis.com/online_prediction_requests_per_base_model`
3. **Or search for:** `imagen-3.0-generate`

---

## 📊 WHAT TO LOOK FOR

The quota you need should have:
- ✅ **Type:** Quota
- ✅ **Dimensions:** `region: us-central1` (or your location)
- ✅ **Base model:** `imagen-3.0-generate` or `imagen-3.0-generate-002`
- ✅ **Value:** Probably 10-20 (current limit)
- ✅ **Adjustable:** Yes

---

## ⚠️ IF YOU CAN'T FIND IT

The quota might be listed under a different name. Try:

1. **Search for:** `prediction` (broader search)
2. **Search for:** `generate` (to find generation quotas)
3. **Check all quotas** with dimension `region: us-central1`
4. **Look for quotas** with value around 10-20 (your current limit)

---

## 💡 QUICK TIP

Since you have 12,039 quotas, **use the filter** - it's essential!  
The filter search is case-insensitive and searches all quota names and descriptions.

---

**Status:** On correct page - Need to filter and find Imagen quota  
**Next:** Filter → Find quota → Request increase











