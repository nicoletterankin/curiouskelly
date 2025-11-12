# Reference Image Format - Current Status

**Date:** November 1, 2025  
**Status:** ⚠️ FORMAT ISSUE - API Not Accepting Reference Images

---

## ✅ QUOTA SUCCESS

**Quota Approved:** ✅ 500 requests/minute  
**Status:** Active and working  
**No quota errors** - API calls succeed without reference images

---

## ⚠️ REFERENCE IMAGE FORMAT ISSUE

**Error:** "No uri or raw bytes are provided in media content"  
**Tried Formats:**
1. ❌ `rawBytes` (base64 string)
2. ❌ `bytesBase64Encoded` (base64 string)
3. ❌ `bytesBase64Encoded` + `mimeType`

**Current Format:**
```json
{
  "referenceImages": [{
    "referenceType": "REFERENCE_TYPE_SUBJECT",
    "referenceId": 1,
    "referenceImage": {
      "bytesBase64Encoded": "BASE64_STRING",
      "mimeType": "image/png"
    },
    "subjectImageConfig": {
      "subjectDescription": "...",
      "subjectType": "SUBJECT_TYPE_PERSON"
    }
  }]
}
```

---

## 🔍 POSSIBLE ISSUES

1. **Predict Endpoint Limitation:** Reference images might not be supported in the `/predict` endpoint
2. **API Version:** Might need different endpoint or API version
3. **Format Structure:** May need different nesting or field names
4. **GCS URI Required:** May need to upload to Google Cloud Storage first

---

## 🎯 WORKAROUND OPTIONS

### Option 1: Use Text Descriptions (Current)
- ✅ Works immediately
- ✅ Character consistency through detailed prompts
- ⚠️ May not achieve perfect likeness

### Option 2: Continue Format Research
- Check official Vertex AI Imagen 3.0 REST API docs
- Try alternative endpoints
- Try GCS URI approach

### Option 3: Use Reference Images in Prompt
- Embed reference images as base64 in prompt text
- Use image-to-image generation if available
- Alternative API methods

---

## ✅ WHAT'S WORKING

1. **Quota:** 500 requests/minute ✅
2. **API Authentication:** OAuth2 working ✅
3. **Asset Generation:** All assets generating successfully ✅
4. **Character Prompts:** Detailed text descriptions working ✅

---

## 📋 NEXT STEPS

### Immediate (Can Do Now)
1. **Regenerate Kelly Assets** with improved text prompts
2. **Generate Missing Assets** for Reinmaker
3. **Validate Assets** using automated validator

### Future (Research Needed)
1. **Research Reference Image Format** - Official Vertex AI docs
2. **Try Alternative Endpoints** - Different API versions
3. **Try GCS URI Approach** - Upload references to Cloud Storage
4. **Contact Google Support** - Ask about reference image format

---

## 💡 RECOMMENDATION

**For Now:** Proceed with text-based generation
- We have detailed character specifications
- Prompts are comprehensive
- Can achieve good character consistency
- Can regenerate later if reference images work

**Future:** Continue reference image research
- Critical for perfect character likeness
- Worth investigating alternative approaches
- May require different API endpoint

---

**Status:** ⚠️ Reference Images Not Working - Text Prompts Active  
**Priority:** MEDIUM - Can proceed with text prompts, investigate reference images separately










