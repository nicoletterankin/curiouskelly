# Reference Image Format Fix Plan

**Date:** November 1, 2025  
**Status:** 🔴 CRITICAL - Fixing Reference Image Format

---

## 🔴 THE PROBLEM

**Error:** "Reference image should have image type"  
**Impact:** Cannot use reference images for character likeness  
**Current Format:** 
```json
{
  "referenceImages": [
    {
      "bytesBase64Encoded": "...",
      "mimeType": "image/png"
    }
  ]
}
```

---

## ✅ ATTEMPTED FIX

### Format Change 1: Wrapped in "image" Object
**New Format:**
```json
{
  "referenceImages": [
    {
      "image": {
        "bytesBase64Encoded": "...",
        "mimeType": "image/png"
      }
    }
  ]
}
```

**Rationale:** Error says "should have image type" - may need "image" wrapper

---

## 🔍 ALTERNATIVE FORMATS TO TRY

### Format 2: "image" at Root Level
```json
{
  "referenceImages": [
    {
      "image": {
        "bytesBase64Encoded": "...",
        "mimeType": "image/png"
      }
    }
  ]
}
```

### Format 3: Direct "image" Field
```json
{
  "instances": [{
    "prompt": "...",
    "referenceImages": [
      {
        "image": {
          "bytesBase64Encoded": "...",
          "mimeType": "image/png"
        }
      }
    ]
  }]
}
```

### Format 4: In Parameters Section
```json
{
  "instances": [{
    "prompt": "..."
  }],
  "parameters": {
    "referenceImages": [
      {
        "image": {
          "bytesBase64Encoded": "...",
          "mimeType": "image/png"
        }
      }
    ]
  }
}
```

### Format 5: Using Google Cloud Storage URI
If base64 doesn't work, may need to upload to GCS first:
```json
{
  "referenceImages": [
    {
      "gcsUri": "gs://bucket/image.png"
    }
  ]
}
```

---

## 🧪 TESTING APPROACH

1. **Test Format 1:** Wrapped format (current attempt)
2. **If fails:** Try Format 2, 3, 4
3. **If all fail:** Research official Vertex AI documentation
4. **Last resort:** Use Google Cloud Storage URIs

---

## 📋 CURRENT IMPLEMENTATION

**Location:** `generate_assets.ps1` → `Generate-VertexAI-Asset` function  
**Line:** ~504-515

**Current Code:**
```powershell
$referenceImageArray += @{
    "image" = @{
        "bytesBase64Encoded" = $refImg.Base64
        "mimeType" = $refImg.MimeType
    }
}
```

---

## 🎯 NEXT STEPS

1. ✅ **Update format** to wrapped "image" object (DONE)
2. **Test** with `test_reference_fix.ps1`
3. **If fails:** Try alternative formats
4. **If succeeds:** Update all generation scripts to use reference images
5. **Regenerate** Kelly assets with working reference images

---

## 📚 RESEARCH NEEDED

- Official Vertex AI Imagen 3.0 REST API documentation
- Reference image format specification
- Example payloads with reference images
- Error message interpretation

---

**Status:** Testing Format 1 (wrapped "image" object)  
**Priority:** HIGH - Character likeness depends on reference images












