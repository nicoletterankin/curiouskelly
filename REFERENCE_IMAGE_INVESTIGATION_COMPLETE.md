# Reference Image Investigation - File by File Analysis Complete

**Date:** November 1, 2025  
**Status:** 🔍 INVESTIGATION COMPLETE - Multiple Format Attempts Documented

---

## 📋 FILES INVESTIGATED

### 1. REFERENCE_IMAGE_CORRECT_FORMAT.md
- **Claims:** Format identified with `rawBytes`
- **Actual Test:** ❌ Failed with `rawBytes`
- **Status:** Documentation may be outdated

### 2. REFERENCE_IMAGE_FORMAT_FIX.md
- **Lists:** Multiple format alternatives
- **Tried:** Wrapped formats, different nesting
- **Status:** All documented formats tested

### 3. REFERENCE_IMAGE_SYSTEM_COMPLETE.md
- **Claims:** Format corrected
- **Reality:** Format still not working
- **Status:** Needs update

### 4. REFERENCE_IMAGE_FORMAT_STATUS.md
- **Status:** Current status document
- **Error:** "No uri or raw bytes are provided in media content"
- **Status:** Accurate current status

### 5. generate_assets.ps1 (Lines 493-525)
- **Current Format:** `rawBytes` + `mimeType` in `referenceImage` object
- **Structure:** REFERENCE_TYPE_SUBJECT with referenceId and subjectImageConfig
- **Status:** ✅ Implemented correctly per documentation

### 6. tools/kelly_asset_generator.py
- **Uses:** Python SDK (`VertexImage.load_from_file()`)
- **Format:** SDK handles encoding internally
- **Key Insight:** SDK vs REST API may have different requirements

### 7. test_reference_fix.ps1
- **Purpose:** Test reference image format
- **Results:** All formats fail with same error
- **Status:** ✅ Test script working

---

## 🔍 KEY FINDINGS

### Finding 1: Field Name Confusion
- Documentation says: `rawBytes`
- We tried: `bytesBase64Encoded` → Failed
- We tried: `rawBytes` → Failed
- **Error:** "No uri or raw bytes are provided in media content"

### Finding 2: Error Message Analysis
- **Error:** "Image editing failed with the following error: No uri or raw bytes are provided in media content"
- **Key Word:** "Image editing" - suggests this might be for editing, not generation
- **Possible Issue:** `/predict` endpoint may not support reference images

### Finding 3: SDK vs REST API
- **Python SDK:** Uses `VertexImage.load_from_file()` - works
- **REST API:** Requires explicit format - failing
- **Implication:** REST API format may be fundamentally different

### Finding 4: Current Payload Structure
```json
{
  "instances": [{
    "prompt": "... [1]",
    "referenceImages": [{
      "referenceType": "REFERENCE_TYPE_SUBJECT",
      "referenceId": 1,
      "referenceImage": {
        "rawBytes": "BASE64_STRING",
        "mimeType": "image/png"
      },
      "subjectImageConfig": {
        "subjectDescription": "...",
        "subjectType": "SUBJECT_TYPE_PERSON"
      }
    }]
  }],
  "parameters": {
    "sampleCount": 1,
    "aspectRatio": "3:4"
  }
}
```

---

## 🧪 FORMATS TESTED (All Failed)

### Format 1: bytesBase64Encoded ✅ Tested
```json
"referenceImage": {
  "bytesBase64Encoded": "..."
}
```
**Result:** ❌ "No uri or raw bytes are provided"

### Format 2: bytesBase64Encoded + mimeType ✅ Tested
```json
"referenceImage": {
  "bytesBase64Encoded": "...",
  "mimeType": "image/png"
}
```
**Result:** ❌ "No uri or raw bytes are provided"

### Format 3: rawBytes ✅ Tested
```json
"referenceImage": {
  "rawBytes": "..."
}
```
**Result:** ❌ "Image should have either uri or image bytes"

### Format 4: rawBytes + mimeType ✅ Tested
```json
"referenceImage": {
  "rawBytes": "...",
  "mimeType": "image/png"
}
```
**Result:** ❌ "Image should have either uri or image bytes"

---

## 🎯 HYPOTHESES

### Hypothesis 1: Endpoint Limitation
**Theory:** `/predict` endpoint may not support reference images  
**Evidence:** Error says "Image editing failed" (suggests editing endpoint)  
**Test:** Try different endpoint or check if reference images only work with SDK

### Hypothesis 2: Field Name Issue
**Theory:** Field name might be "imageBytes" or "bytes" not "rawBytes"  
**Evidence:** Error says "image bytes" not "raw bytes"  
**Test:** Try "imageBytes" or "bytes" field name

### Hypothesis 3: Decoded Bytes Required
**Theory:** API might want decoded bytes array, not base64 string  
**Evidence:** Error says "image bytes" - might mean decoded  
**Test:** Send decoded byte array instead of base64 string

### Hypothesis 4: GCS URI Required
**Theory:** REST API might require GCS URIs, not base64  
**Evidence:** Error mentions "uri" as alternative  
**Test:** Upload reference images to GCS and use gcsUri

### Hypothesis 5: Different API Version
**Theory:** Reference images might only work with newer API version  
**Evidence:** Current model: `imagen-3.0-generate-002`  
**Test:** Try different model version or endpoint

---

## 🔬 NEXT TESTS TO TRY

### Test 1: Try "imageBytes" Field Name
```json
"referenceImage": {
  "imageBytes": "BASE64_STRING"
}
```

### Test 2: Try "bytes" Field Name
```json
"referenceImage": {
  "bytes": "BASE64_STRING"
}
```

### Test 3: Try Decoded Bytes Array
```powershell
# Convert base64 to byte array
$bytes = [System.Convert]::FromBase64String($refImg.Base64)
"referenceImage": {
  "bytes": $bytes  # Array of bytes, not string
}
```

### Test 4: Try GCS URI Format
```json
"referenceImage": {
  "gcsUri": "gs://bucket-name/image.png"
}
```

### Test 5: Check Different Endpoint
- Try `/generateImages` instead of `/predict`
- Try different API version
- Check if reference images only work with SDK

---

## 📊 COMPARISON: SDK vs REST API

| Aspect | Python SDK | REST API |
|--------|------------|----------|
| Image Loading | `VertexImage.load_from_file(path)` | Manual base64 encoding |
| Format | SDK handles internally | Must specify format |
| Reference Images | `params["reference_images"] = ref_images` | `referenceImages` array |
| Status | ✅ Works | ❌ Format unknown |

---

## 💡 RECOMMENDATION

**Option A: Use Python SDK** (If Possible)
- Python SDK handles reference images correctly
- Would require switching from PowerShell to Python
- More reliable but requires Python environment

**Option B: Continue REST API Research**
- Find official REST API documentation
- Try GCS URI approach
- Contact Google Support for format specification

**Option C: Proceed Without Reference Images**
- Use enhanced text prompts
- Achieve good character consistency
- Regenerate later if reference images work

---

## 📝 INVESTIGATION SUMMARY

**Files Reviewed:** 7  
**Formats Tested:** 4  
**All Failed:** ✅ Yes  
**Error Consistent:** "No uri or raw bytes" / "Image should have either uri or image bytes"  
**Next Step:** Try GCS URI or contact Google Support

---

**Status:** 🔍 Investigation Complete - Ready for Next Tests  
**Priority:** HIGH - Critical for character consistency











