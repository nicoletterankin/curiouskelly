# Cloudflare R2 Setup Guide for Kelly Assets

This guide walks through setting up the `kelly-assets` R2 bucket and configuring access.

## Step 1: Create R2 Bucket

1. Go to Cloudflare Dashboard: https://dash.cloudflare.com/
2. Navigate to **R2 Object Storage**
3. Click **Create bucket**
4. Name: `kelly-assets`
5. Location: **Automatic** (or choose closest to your users)
6. Click **Create bucket**

## Step 2: Create Folder Structure

You can create folders by uploading a placeholder file or using the Cloudflare API. Here's the structure:

```
kelly-assets/
├── production/
│   ├── poses/
│   │   ├── idle/
│   │   ├── thinking/
│   │   ├── pointing-left/
│   │   ├── pointing-right/
│   │   ├── pointing-up/
│   │   ├── pointing-down/
│   │   ├── encouraging/
│   │   ├── hint/
│   │   ├── celebrating/
│   │   ├── supportive/
│   │   ├── proud/
│   │   └── excited/
│   └── reference/
├── staging/
├── review/
├── archive/
└── lora-training/
    └── dataset/
```

## Step 3: Create R2 API Token

1. In R2 dashboard, click **Manage R2 API Tokens**
2. Click **Create API token**
3. Name: `kelly-assets-access`
4. Permissions:
   - **Object Read & Write**
   - Scope: Apply to specific buckets only
   - Select: `kelly-assets`
5. Click **Create API Token**
6. **IMPORTANT:** Copy the Access Key ID and Secret Access Key immediately

## Step 4: Configure Custom Domain (Optional but Recommended)

1. In the `kelly-assets` bucket settings, click **Settings**
2. Scroll to **Public access**
3. Click **Connect domain**
4. Enter: `kelly-assets.curiouskelly.com`
5. Add the CNAME record to your DNS:
   ```
   CNAME kelly-assets -> [your-r2-bucket-url]
   ```
6. Wait for DNS propagation (~5-10 minutes)

## Step 5: Set CORS Policy

To allow web access to images:

1. In bucket settings, go to **CORS policy**
2. Add this policy:

```json
[
  {
    "AllowedOrigins": [
      "https://curiouskelly.com",
      "https://*.curiouskelly.com",
      "http://localhost:*"
    ],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

## Step 6: Update Environment Variables

Add to `.env.local`:

```env
# Cloudflare R2 Configuration
CLOUDFLARE_ACCOUNT_ID=your-cloudflare-account-id
CLOUDFLARE_R2_ACCESS_KEY_ID=[your-access-key-id]
CLOUDFLARE_R2_SECRET_ACCESS_KEY=[your-secret-access-key]
KELLY_ASSETS_BUCKET=kelly-assets
KELLY_ASSETS_CDN_URL=https://kelly-assets.curiouskelly.com

# Google AI Studio (for image generation)
GOOGLE_AI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

## Step 7: Test Upload Script

Create `scripts/kelly-visual-identity/test-r2-upload.ts`:

```typescript
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import * as fs from "fs";

const s3 = new S3Client({
  region: "auto",
  endpoint: `https://${process.env.CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com`,
  credentials: {
    accessKeyId: process.env.CLOUDFLARE_R2_ACCESS_KEY_ID!,
    secretAccessKey: process.env.CLOUDFLARE_R2_SECRET_ACCESS_KEY!,
  },
});

async function testUpload() {
  const testContent = "Kelly Assets Test File";
  
  await s3.send(new PutObjectCommand({
    Bucket: "kelly-assets",
    Key: "test/hello.txt",
    Body: testContent,
    ContentType: "text/plain",
  }));
  
  console.log("✅ Upload successful!");
  console.log(`📍 URL: https://kelly-assets.curiouskelly.com/test/hello.txt`);
}

testUpload();
```

Run: `tsx scripts/kelly-visual-identity/test-r2-upload.ts`

## Step 8: Configure Cloudflare Image Resizing

1. Go to **Speed** → **Optimization** → **Image Resizing**
2. Enable **Image Resizing**
3. This allows URLs like:
   ```
   https://curiouskelly.com/cdn-cgi/image/width=400,quality=85/kelly/poses/idle/kelly_idle_hero.png
   ```

## Next Steps

1. Run the LoRA dataset preparation script
2. Upload reference images to `lora-training/dataset/`
3. Start Civitai training
4. Generate poses using the generation script
5. Upload generated poses to `production/poses/[pose-name]/`
6. Update Supabase with asset metadata

## Troubleshooting

### "Access Denied" errors
- Check that API token has correct permissions
- Verify bucket name matches exactly
- Ensure credentials are in `.env.local`

### CORS errors in browser
- Verify CORS policy is set correctly
- Check that origin matches your domain
- Clear browser cache

### Images not loading
- Check custom domain DNS propagation
- Verify file paths match exactly
- Check bucket public access settings

## Cost Estimates

R2 Pricing (as of 2025):
- **Storage:** $0.015/GB/month
- **Class A operations (writes):** $4.50/million
- **Class B operations (reads):** $0.36/million
- **Egress:** FREE (unlike S3!)

For Kelly assets:
- ~500 images × 2MB average = 1GB storage = **$0.015/month**
- 100K monthly views = **$0.036/month**
- **Total: ~$0.05/month** 🎉

## Security Notes

- Never commit R2 credentials to git
- Use `.env.local` for local development
- Use Vercel/Cloudflare secrets for production
- Rotate API tokens every 90 days
- Monitor access logs for unusual activity






