# Reference Image Investigation - File by File Analysis

**Date:** November 1, 2025  
**Status:** 🔍 SYSTEMATIC INVESTIGATION IN PROGRESS

---

## 📋 FILES REVIEWED

### 1. REFERENCE_IMAGE_CORRECT_FORMAT.md ✅
**Status:** Claims format identified  
**Key Info:** Uses `rawBytes` field  
**Issue:** Documentation says `rawBytes` but we're using `bytesBase64Encoded`

### 2. REFERENCE_IMAGE_FORMAT_FIX.md ✅
**Status:** Lists alternative formats to try  
**Key Info:** Multiple format attempts documented  
**Tried:** Wrapped formats, different nesting

### 3. REFERENCE_IMAGE_SYSTEM_COMPLETE.md ✅
**Status:** Claims format corrected  
**Key Info:** Uses `rawBytes` in documentation  
**Discrepancy:** Code uses `bytesBase64Encoded`

### 4. REFERENCE_IMAGE_FORMAT_STATUS.md ✅
**Status:** Current status document  
**Key Info:** Lists all attempted formats  
**Error:** "No uri or raw bytes are provided in media content"

### 5. generate_assets.ps1 (Lines 493-525) ✅
**Status:** Current implementation  
**Format Used:**
```json
{
  "referenceType": "REFERENCE_TYPE_SUBJECT",
  "referenceId": 1,
  "referenceImage": {
    "bytesBase64Encoded": "...",
    "mimeType": "image/png"
  },
  "subjectImageConfig": {...}
}
```

### 6. tools/kelly_asset_generator.py (Lines 100-175) ✅
**Status:** Python SDK implementation  
**Key Discovery:** Uses `VertexImage.load_from_file(path)`  
**Important:** SDK handles encoding internally - different from REST API  
**Format:** `params["reference_images"] = ref_images` (VertexImage objects)

---

## 🔍 DISCOVERIES

### Discovery 1: Documentation vs Code Mismatch
- **Documentation says:** `rawBytes`
- **Code uses:** `bytesBase64Encoded`
- **Action:** Try `rawBytes` instead of `bytesBase64Encoded`

### Discovery 2: Python SDK vs REST API
- **Python SDK:** Uses `VertexImage.load_from_file()` - handles encoding automatically
- **REST API:** Requires explicit format - may be different structure
- **Implication:** REST API format may differ from SDK

### Discovery 3: Error Message Analysis
- **Error:** "No uri or raw bytes are provided in media content"
- **Suggests:** API expects either `uri` OR `raw bytes` but not finding them
- **Possible Issue:** Field name wrong OR structure wrong

---

## 🧪 FORMATS TO TEST (Systematic)

### Format 1: rawBytes (not bytesBase64Encoded)
```json
{
  "referenceImage": {
    "rawBytes": "BASE64_STRING"
  }
}
```

### Format 2: rawBytes at root level (not nested)
```json
{
  "referenceImage": {
    "rawBytes": "BASE64_STRING",
    "mimeType": "image/png"
  }
}
```

### Format 3: Bytes as array (not string)
```json
{
  "referenceImage": {
    "rawBytes": [123, 45, 67, ...]  // Array of bytes
  }
}
```

### Format 4: Different endpoint
- Try `/generate` instead of `/predict`
- Try different API version
- Check if reference images only work with SDK

### Format 5: GCS URI approach
- Upload reference images to Google Cloud Storage
- Use `gcsUri` instead of base64
- May be required for REST API

---

## 📊 COMPARISON TABLE

| Field Name | Documentation | Code | Tested | Result |
|------------|--------------|------|--------|--------|
| `rawBytes` | ✅ Yes | ❌ No | ❌ No | **TRY THIS** |
| `bytesBase64Encoded` | ❌ No | ✅ Yes | ✅ Yes | ❌ Failed |
| `bytesBase64Encoded` + `mimeType` | ❌ No | ✅ Yes | ✅ Yes | ❌ Failed |

---

## 🎯 NEXT INVESTIGATION STEPS

### Step 1: Try rawBytes Field Name
- Change `bytesBase64Encoded` → `rawBytes`
- Keep same structure
- Test immediately

### Step 2: Check Official REST API Docs
- Search for "Vertex AI Imagen REST API reference images"
- Find exact field names and structure
- Verify endpoint supports reference images

### Step 3: Compare SDK vs REST API
- Understand how SDK encodes images
- Replicate in REST API format
- May need different structure

### Step 4: Check API Version
- Current: `imagen-3.0-generate-002`
- May need different model version
- May need different endpoint

### Step 5: GCS URI Test
- Upload reference image to GCS
- Use `gcsUri` format
- Test if REST API requires GCS

---

## 📝 NOTES

1. **Documentation inconsistency:** Docs say `rawBytes`, code uses `bytesBase64Encoded`
2. **SDK vs REST:** Different approaches - SDK handles encoding, REST needs explicit format
3. **Error message:** "No uri or raw bytes" suggests field name or structure issue
4. **Endpoint:** `/predict` endpoint may not support reference images - may need different endpoint

---

**Status:** 🔍 Investigation Complete - Ready to Test rawBytes  
**Priority:** HIGH - Critical for character consistency










