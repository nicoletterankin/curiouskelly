# Unity WebGL CDN Setup Guide

This document describes how Unity WebGL builds are hosted and served for Curious Kelly.

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────────────────┐     ┌─────────────────┐
│   Browser       │────▶│  Cloudflare Worker (CDN)    │────▶│  Cloudflare R2  │
│   learn.html    │     │  unity-cdn.workers.dev      │     │  (Object Store) │
└─────────────────┘     └─────────────────────────────┘     └─────────────────┘
                                     │
                        Adds CORS headers +
                        Content-Encoding: br
```

## Components

### 1. Cloudflare R2 Bucket

- **Bucket Name:** `curious-kelly-unity`
- **Public URL:** `https://pub-95ad3557cf944f3ea28696e43ddfe4b3.r2.dev`
- **Contents:** Brotli-compressed Unity WebGL build files

**Files stored:**
- `Kelly_Web_Build.loader.js` - Unity loader script (uncompressed)
- `Kelly_Web_Build.data.br` - Game data (Brotli compressed)
- `Kelly_Web_Build.framework.js.br` - Unity framework (Brotli compressed)
- `Kelly_Web_Build.wasm.br` - WebAssembly binary (Brotli compressed)

### 2. Cloudflare Worker

- **Worker Name:** `unity-cdn`
- **URL:** `https://unity-cdn.nicoletterankin.workers.dev`
- **Source:** `infrastructure/cloudflare/unity-cdn-worker/`

**Features:**
- Proxies requests to R2 bucket
- Adds CORS headers for cross-origin requests
- Adds `Content-Encoding: br` for Brotli files
- Aggressive caching (1 year for versioned files)
- Supports versioned paths (`/v1/`, `/v2/`, etc.)

### 3. Unity Loader Script

- **Path:** `public/js/unity-kelly-loader.js`
- **Class:** `UnityKellyLoader`

**Features:**
- CDN-first loading with fallback strategy
- Automatic environment detection (local vs production)
- Progress tracking and error handling
- Performance monitoring integration

## Deployment Workflow

### Manual Upload (Current)

1. Build Unity WebGL in Unity Editor
2. Run compression script:
   ```powershell
   .\scripts\unity\compress-unity-build.ps1
   ```
3. Upload to R2:
   ```powershell
   .\scripts\unity\upload-to-r2.ps1 -Version "v1.0.0"
   ```

### Automated CI/CD (GitHub Actions)

Trigger: Push to `main` with changes to `digital-kelly/engines/Kelly_Engine_V2/onlykelly/`

Workflow: `.github/workflows/unity-build-deploy.yml`

1. Checks out repository
2. Builds Unity WebGL using game-ci/unity-builder
3. Compresses with Brotli
4. Uploads to R2 with versioning

**Required Secrets:**
- `UNITY_LICENSE` - Unity license file content
- `UNITY_EMAIL` - Unity account email
- `UNITY_PASSWORD` - Unity account password
- `CLOUDFLARE_API_TOKEN` - Cloudflare API token with R2 permissions
- `CLOUDFLARE_ACCOUNT_ID` - Cloudflare account ID

## Fallback Strategy

The loader tries multiple sources in order:

**Production (curiouskelly.com):**
1. Cloudflare Worker CDN (Brotli compressed)
2. Local bundled files (if available)
3. Netlify deployment (legacy)

**Development (localhost):**
1. Local build (`/unity/kelly-live/Build/`)
2. Cloudflare Worker CDN
3. Netlify deployment

## File Structure

```
public/
├── js/
│   ├── unity-kelly-loader.js     # Main loader class
│   └── unity-performance-monitor.js  # Performance tracking
└── unity/
    └── kelly-live/
        └── Build/
            ├── WebGL.loader.js    # Local loader
            ├── WebGL.data         # Local data (uncompressed)
            ├── WebGL.framework.js # Local framework
            └── WebGL.wasm         # Local WebAssembly

infrastructure/
└── cloudflare/
    └── unity-cdn-worker/
        ├── src/index.js           # Worker source
        ├── wrangler.toml          # Worker config
        └── README.md              # Worker documentation

scripts/
└── unity/
    ├── compress-unity-build.ps1   # Windows compression
    ├── compress-unity-build.sh    # Linux/Mac compression
    ├── upload-to-r2.ps1           # Windows upload
    └── upload-to-r2.sh            # Linux/Mac upload

.github/
└── workflows/
    ├── unity-build-deploy.yml     # CI/CD pipeline
    └── unity-cdn-health.yml       # Health monitoring
```

## Troubleshooting

### CORS Errors

If you see CORS errors in the console:

1. Check Worker is deployed: `https://unity-cdn.nicoletterankin.workers.dev/`
2. Verify CORS headers:
   ```bash
   curl -I https://unity-cdn.nicoletterankin.workers.dev/Kelly_Web_Build.loader.js
   ```
3. Should see: `Access-Control-Allow-Origin: *`

### Brotli Decompression Fails

If browser can't decompress `.br` files:

1. Check Content-Encoding header:
   ```bash
   curl -I https://unity-cdn.nicoletterankin.workers.dev/Kelly_Web_Build.wasm.br
   ```
2. Should see: `Content-Encoding: br`

### Unity Instance Not Created

If `createUnityInstance is not defined`:

1. Check loader script loaded successfully
2. Check browser console for 404 errors
3. Verify file exists in R2:
   ```bash
   wrangler r2 object head curious-kelly-unity Kelly_Web_Build.loader.js
   ```

### Performance Issues

1. Check load metrics:
   ```javascript
   window.unityKellyLoader.printPerformanceReport();
   ```
2. Check network tab for slow requests
3. Ensure Brotli files are being used (much smaller than uncompressed)

## Versioning

Files can be uploaded with version prefixes:

```
curious-kelly-unity/
├── Kelly_Web_Build.loader.js      # Latest
├── Kelly_Web_Build.data.br        # Latest
├── v1/
│   ├── Kelly_Web_Build.loader.js  # Version 1
│   └── ...
└── v2/
    ├── Kelly_Web_Build.loader.js  # Version 2
    └── ...
```

To use a specific version, modify the loader's CDN URL:
```javascript
window.unityKellyLoader = new UnityKellyLoader({
  cdnUrl: 'https://unity-cdn.nicoletterankin.workers.dev/v2'
});
```

## Health Monitoring

The health check workflow runs every 6 hours:
- Verifies all files are accessible
- Checks CORS headers
- Checks Content-Encoding headers
- Measures load times

View results in GitHub Actions: `.github/workflows/unity-cdn-health.yml`

## Cost Considerations

**Cloudflare R2:**
- Storage: $0.015/GB/month
- Class A operations (writes): $4.50/million
- Class B operations (reads): $0.36/million
- No egress fees!

**Current Unity build size (compressed):**
- ~15-30 MB total
- Monthly cost estimate: < $1 for typical usage

## Security

- R2 bucket is public read-only
- Worker validates origin for CORS
- Files are immutable (versioned uploads)
- No sensitive data in Unity build




