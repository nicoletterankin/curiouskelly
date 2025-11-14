# Google Cloud Vertex AI Quota Upgrade Guide

**Project ID:** `gen-lang-client-0005524332`  
**Service:** Vertex AI Imagen 3.0  
**Quota Type:** `aiplatform.googleapis.com/online_prediction_requests_per_base_model`  
**Model:** `imagen-3.0-generate`

---

## 🚀 QUICK UPGRADE OPTIONS

### Option 1: Google Cloud Console (Web UI) - RECOMMENDED

1. **Go to Quotas Page:**
   - Open: https://console.cloud.google.com/apis/api/aiplatform.googleapis.com/quotas?project=gen-lang-client-0005524332
   - Or navigate: **APIs & Services → Quotas** in Google Cloud Console

2. **Filter for Imagen:**
   - Search for: `imagen-3.0-generate`
   - Or filter by: `aiplatform.googleapis.com/online_prediction_requests_per_base_model`

3. **Request Increase:**
   - Click the quota you want to increase
   - Click **"EDIT QUOTAS"** button
   - Enter new limit (e.g., 100, 500, or 1000 requests per minute)
   - Provide justification: "Generating game assets for production - need higher throughput for batch image generation"
   - Submit request

4. **Wait for Approval:**
   - Usually approved within 24-48 hours
   - You'll receive email when approved

---

### Option 2: Direct Quota Request Link

**Direct Link (if logged in):**
https://console.cloud.google.com/apis/api/aiplatform.googleapis.com/quotas?project=gen-lang-client-0005524332&pageState=(%22allQuotasTable%22:(%22f%22:%22%255B%257B_22k_22_3A_22_22_2C_22t_22_3A10_2C_22v_22_3A_22_2Fapis%2Fapi%2Faiplatform.googleapis.com%2Fquotas%22_2C_22s_22_3Atrue_2C_22i_22_3A_22online_prediction_requests_per_base_model%22%257D%255D%22))

---

### Option 3: gcloud CLI Command

```powershell
# List current quotas
gcloud alpha services quota list \
  --service=aiplatform.googleapis.com \
  --consumer=projects/gen-lang-client-0005524332 \
  --filter="metric=aiplatform.googleapis.com/online_prediction_requests_per_base_model"

# Request quota increase (opens web form)
gcloud alpha services quota update \
  --service=aiplatform.googleapis.com \
  --consumer=projects/gen-lang-client-0005524332 \
  --metric=aiplatform.googleapis.com/online_prediction_requests_per_base_model \
  --value=100
```

---

## 📋 RECOMMENDED QUOTA VALUES

**For Development/Testing:**
- Current: Likely ~10-20 requests/minute
- Recommended: **50-100 requests/minute**

**For Production/Asset Generation:**
- Recommended: **200-500 requests/minute**
- For batch generation: **1000+ requests/minute**

---

## 💡 QUOTA TYPES TO REQUEST

1. **`online_prediction_requests_per_base_model`** (Primary)
   - Limits: Requests per minute to Imagen model
   - **This is what you're hitting!**

2. **`online_prediction_requests`** (Secondary)
   - Total requests across all Vertex AI models
   - May also need increase if generating many assets

---

## ⏱️ WAITING TIME

- **Auto-approved:** Usually for small increases (< 2x current limit)
- **Manual review:** 24-48 hours for larger increases
- **Enterprise:** May require sales contact for very high limits

---

## 🔍 CHECK CURRENT QUOTA

```powershell
# Check current quota usage
gcloud alpha services quota list \
  --service=aiplatform.googleapis.com \
  --consumer=projects/gen-lang-client-0005524332
```

---

## 📝 JUSTIFICATION TEMPLATE

**Use this when requesting increase:**

```
Project: gen-lang-client-0005524332
Service: Vertex AI Imagen 3.0
Requested Limit: 100 requests/minute

Justification:
We are generating game assets for production use. Our workflow requires:
- Batch generation of character assets (50+ images)
- Reference image-based generation (higher API usage per request)
- Quality assurance iterations
- Production asset pipeline

Current limit is insufficient for our development and production needs.
```

---

## 🎯 IMMEDIATE WORKAROUND

**While waiting for quota approval:**

1. **Batch with delays:** Add 60-second delays between requests
2. **Off-peak hours:** Run generation during off-peak times
3. **Reduce batch size:** Generate fewer assets per run
4. **Use existing assets:** Focus on regenerating only critical assets

---

## ✅ NEXT STEPS

1. **Request quota increase** using Option 1 (Console) - EASIEST
2. **Wait for approval** (usually 24-48 hours)
3. **Test reference images** once quota is increased
4. **Regenerate all Kelly assets** with working reference images

---

**Status:** Ready to upgrade  
**Priority:** HIGH - Blocking reference image testing











