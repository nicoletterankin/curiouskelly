# Reference Image Format - FINAL STATUS & WORKAROUND

**Date:** November 1, 2025  
**Status:** ⚠️ REST API LIMITATION IDENTIFIED

---

## 🔴 THE PROBLEM

**Error:** "No uri or raw bytes are provided in media content"  
**Root Cause:** The `/predict` endpoint for Vertex AI Imagen 3.0 generation may not support reference images via REST API

**Evidence:**
- ✅ Python SDK works (`VertexImage.load_from_file()`)
- ❌ REST API `/predict` endpoint fails with all formats
- Error suggests API expects different structure or endpoint

---

## ✅ WHAT WE'VE TRIED

### Format Attempts (All Failed):
1. ❌ `bytesBase64Encoded` alone
2. ❌ `bytesBase64Encoded` + `mimeType`
3. ❌ `rawBytes` alone
4. ❌ `rawBytes` + `mimeType`
5. ❌ `bytes` field name
6. ❌ Flattened structure
7. ❌ Nested `referenceImage` object

### Error Messages:
- "No uri or raw bytes are provided in media content"
- "Image should have either uri or image bytes"
- "Image editing failed" (suggests endpoint mismatch)

---

## 💡 WORKAROUND: Use Python SDK

- ✅ **Implemented:** `tools/generate_vertex_image_with_references.py` helper script
  - Uses Vertex AI Python SDK (`VertexImage.load_from_file`) for true reference control
  - Automatically invoked by `generate_assets.ps1` when reference images are present
  - Falls back to REST API + enhanced text prompts if Python execution fails

### Option 1: Python SDK Helper (Preferred)
1. Ensure `google-cloud-aiplatform` and `pillow` are installed in the Python environment
2. Confirm `GOOGLE_CLOUD_PROJECT` env var (or pass `--project`) and gcloud auth
3. Script usage example:
   ```bash
   python tools/generate_vertex_image_with_references.py \
     --prompt "Kelly in Reinmaker armor [1]" \
     --negative-prompt "cartoon, stylized" \
     --aspect-ratio 3:4 \
     --output outputs/kelly_test.png \
     --reference Ref/headshot2-kelly-base169\ 101225.png \
     --reference Ref/kelly_directors_chair_8k_light\ (2).png \
     --width 1024 --height 1280
   ```

### Option 2: Continue with Enhanced Text Prompts
- Use detailed character specifications
- Achieve good character consistency
- Regenerate later if reference images work

---

## 📋 CURRENT WORKING SOLUTION

**For Now:** Use enhanced text prompts
- ✅ Detailed character specifications implemented
- ✅ Hair and face descriptions comprehensive
- ✅ Can achieve good character consistency
- ✅ Working immediately

**Future:** Investigate GCS URI approach or different endpoint

---

## 🎯 NEXT STEPS

1. **Immediate:** Proceed with text-based generation
2. **Future:** Try GCS URI format (upload references to Cloud Storage)
3. **Future:** Check if different endpoint supports reference images
4. **Future:** Consider Python SDK wrapper for critical assets

---

**Status:** ⚠️ REST API Limitation - Using Text Prompts  
**Priority:** MEDIUM - Text prompts working, reference images future enhancement

