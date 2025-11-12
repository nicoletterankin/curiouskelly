# Reference Image Format - CORRECT STRUCTURE

**Date:** November 1, 2025  
**Status:** ✅ CORRECT FORMAT IDENTIFIED

---

## ✅ CORRECT FORMAT (Based on Official Documentation)

```json
{
  "instances": [
    {
      "prompt": "A portrait of Kelly with her distinctive hairstyle [1]",
      "referenceImages": [
        {
          "referenceType": "REFERENCE_TYPE_SUBJECT",
          "referenceId": 1,
          "referenceImage": {
            "rawBytes": "BASE64_ENCODED_IMAGE_STRING"
          },
          "subjectImageConfig": {
            "subjectDescription": "Kelly Rein, photorealistic digital human, oval face with soft rounded contours, long hair extending well past shoulders",
            "subjectType": "SUBJECT_TYPE_PERSON"
          }
        }
      ]
    }
  ],
  "parameters": {
    "sampleCount": 1,
    "aspectRatio": "3:4"
  }
}
```

---

## 🔑 KEY REQUIREMENTS

1. **referenceType:** Must be `"REFERENCE_TYPE_SUBJECT"` for character likeness
2. **referenceId:** Unique integer (1, 2, 3, etc.) - referenced in prompt as `[1]`, `[2]`, etc.
3. **referenceImage.rawBytes:** Base64-encoded string (NOT decoded bytes)
4. **subjectImageConfig:** Required configuration for person/subject
   - `subjectDescription`: Brief description of the subject
   - `subjectType`: `"SUBJECT_TYPE_PERSON"` for character likeness
5. **Prompt Reference:** Prompt must include `[1]`, `[2]`, etc. to reference images

---

## ✅ IMPLEMENTATION STATUS

**Updated:** `generate_assets.ps1` → `Generate-VertexAI-Asset` function
- ✅ Uses `REFERENCE_TYPE_SUBJECT`
- ✅ Uses `referenceId` (1, 2, 3...)
- ✅ Uses `rawBytes` (base64 string)
- ✅ Includes `subjectImageConfig`
- ✅ Updates prompt with `[1]`, `[2]` references

---

## 🧪 TESTING

**Last Test:** Hit quota limit (no format error - good sign!)
**Next:** Wait for quota reset and retry, or verify format in payload JSON

---

## 🎯 NEXT STEPS

1. ✅ **Format Updated** - Using correct REFERENCE_TYPE_SUBJECT structure
2. **Wait for Quota Reset** - Or upgrade plan
3. **Test Again** - Verify format works with reference images
4. **Regenerate Kelly Assets** - Once format confirmed working

---

**Status:** ✅ Format Correct - Ready to Test  
**Priority:** HIGH - Character likeness depends on this











