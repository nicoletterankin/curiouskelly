Here is a reproduction curl for the HeyGen team to verify that the specific Talking Photo IDs are accessible and working with v2 generation.

**Message to HeyGen:**
"Hi Team, we are preparing a high-volume launch and need to confirm that our specific Talking Photo IDs are fully propagated and accessible via the API. We are using the `v2/video/generate` endpoint.

Here are the details:
- **Group ID:** `a762125d3107477aba43d1bd79f90d6e`
- **Sample Avatar ID (The Architect):** `06b78109ad22489ea2165ebbf180f77b`

Below is the CURL command we are using. Could you please run this against your backend logs/test to confirm visibility?"

```bash
curl -X POST "https://api.heygen.com/v2/video/generate" \
  -H "X-Api-Key: <YOUR_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "video_inputs": [
      {
        "character": {
          "type": "talking_photo",
          "talking_photo_id": "06b78109ad22489ea2165ebbf180f77b"
        },
        "voice": {
          "type": "text",
          "input_text": "This is a connectivity test for the Curious Kelly Architect avatar."
        },
        "background": {
          "type": "color",
          "value": "#FFFFFF"
        }
      }
    ],
    "dimension": {
      "width": 1080,
      "height": 1080
    },
    "test": true
  }'
```

**List Verification Hook:**
(To verify the specific ID exists in the list)
```bash
curl -X GET "https://api.heygen.com/v1/talking_photo.list" \
  -H "X-Api-Key: <YOUR_API_KEY>"
```









