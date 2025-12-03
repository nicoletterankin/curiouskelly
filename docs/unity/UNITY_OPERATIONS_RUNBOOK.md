# Unity CDN Operations Runbook

Quick reference for common operations and incident response.

## Quick Commands

### Check CDN Health

```bash
# Test loader endpoint
curl -I https://unity-cdn.nicoletterankin.workers.dev/Kelly_Web_Build.loader.js

# Test Brotli file with encoding
curl -I -H "Accept-Encoding: br" https://unity-cdn.nicoletterankin.workers.dev/Kelly_Web_Build.wasm.br

# Full health check
curl -s https://unity-cdn.nicoletterankin.workers.dev/Kelly_Web_Build.loader.js | head -c 100
```

### List R2 Bucket Contents

```bash
wrangler r2 object list curious-kelly-unity --max-keys=20
```

### Upload New Build

```powershell
# Windows
cd scripts\unity
.\compress-unity-build.ps1
.\upload-to-r2.ps1 -Version "v$(Get-Date -Format 'yyyyMMdd')"
```

```bash
# Linux/Mac
cd scripts/unity
./compress-unity-build.sh
./upload-to-r2.sh "" "v$(date +%Y%m%d)"
```

### Deploy Worker Update

```bash
cd infrastructure/cloudflare/unity-cdn-worker
wrangler deploy
```

## Incident Response

### 🔴 Unity Not Loading

**Symptoms:** 3D mode button shows error, console shows failed fetch requests

**Immediate Actions:**

1. Check CDN status:
   ```bash
   curl -I https://unity-cdn.nicoletterankin.workers.dev/Kelly_Web_Build.loader.js
   ```

2. If 404 - Files missing from R2:
   - Re-upload build files
   - Check R2 bucket exists

3. If 500 - Worker error:
   - Check Cloudflare dashboard for worker logs
   - Redeploy worker: `wrangler deploy`

4. If CORS error:
   - Verify headers present
   - Check allowed origins in worker

**Recovery:**
- Loader has automatic fallback to Netlify
- If CDN is down, users will see slower loads but should still work

### 🟡 Slow Load Times

**Symptoms:** 3D mode takes > 30 seconds to load

**Investigation:**

1. Check load metrics in browser:
   ```javascript
   window.unityKellyLoader.printPerformanceReport();
   ```

2. Check file sizes:
   ```bash
   wrangler r2 object head curious-kelly-unity Kelly_Web_Build.data.br
   ```

3. Check if Brotli is being used (Content-Encoding header)

**Mitigation:**
- Ensure Brotli files are uploaded (not uncompressed)
- Check user's network connection
- Consider regional CDN caching issues

### 🟡 WebGL Errors in Console

**Symptoms:** "WebGL: INVALID_ENUM" or shader errors

**Context:** These are often GPU compatibility warnings, not failures

**Actions:**

1. Check if Unity actually loaded:
   ```javascript
   window.unityKellyLoader.isLoaded // should be true
   ```

2. If loaded successfully, warnings can be ignored

3. If rendering issues visible:
   - Try different browser
   - Update GPU drivers
   - Check WebGL compatibility: https://get.webgl.org/

### 🔴 R2 Bucket Access Denied

**Symptoms:** 403 errors when accessing R2

**Actions:**

1. Check bucket exists:
   ```bash
   wrangler r2 bucket list
   ```

2. Check bucket permissions in Cloudflare dashboard

3. Re-enable public access if disabled

4. Check Worker's R2 binding in `wrangler.toml`

## Deployment Checklist

### Before Deploying Unity Build

- [ ] Test locally with uncompressed files first
- [ ] Verify Unity build completes without errors
- [ ] Run compression script
- [ ] Verify Brotli files exist
- [ ] Test compressed files load locally

### After Deploying to R2

- [ ] Verify files uploaded: `wrangler r2 object list curious-kelly-unity`
- [ ] Test CDN endpoint responds
- [ ] Test CORS headers present
- [ ] Test Content-Encoding header for .br files
- [ ] Load production site and test 3D mode
- [ ] Check browser console for errors

## Monitoring

### Automated Health Checks

- GitHub Actions: `.github/workflows/unity-cdn-health.yml`
- Runs every 6 hours
- Checks: file availability, CORS, Content-Encoding, load times

### Manual Health Check

```bash
# Run local health check script
curl -s -I https://unity-cdn.nicoletterankin.workers.dev/Kelly_Web_Build.loader.js | grep -E "HTTP|Access-Control|Content-"
```

### Cloudflare Dashboard

- Worker analytics: Requests, errors, latency
- R2 metrics: Storage, bandwidth, operations

## Contacts

- **Cloudflare Account:** [Cloudflare Dashboard](https://dash.cloudflare.com)
- **GitHub Actions:** [Actions Tab](../../actions)
- **Documentation:** [UNITY_CDN_SETUP.md](./UNITY_CDN_SETUP.md)

## Appendix: Environment Variables

### GitHub Secrets Required

| Secret | Description |
|--------|-------------|
| `UNITY_LICENSE` | Unity license file (base64) |
| `UNITY_EMAIL` | Unity account email |
| `UNITY_PASSWORD` | Unity account password |
| `CLOUDFLARE_API_TOKEN` | API token with R2 write access |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account ID |

### Local Development

Create `.env` file (do not commit):
```
CLOUDFLARE_API_TOKEN=your_token
CLOUDFLARE_ACCOUNT_ID=your_account_id
```

Or use `wrangler login` for interactive auth.

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-12-03 | Initial CDN setup with Cloudflare Worker | AI Assistant |
| 2025-12-03 | Added CI/CD pipeline and health checks | AI Assistant |

