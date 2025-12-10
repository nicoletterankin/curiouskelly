# Unity CDN Worker

Cloudflare Worker that serves Unity WebGL build assets from R2 with proper headers.

## Features

- **CORS Headers**: Allows requests from curiouskelly.com and localhost
- **Content-Encoding**: Sets `Content-Encoding: br` for Brotli-compressed files
- **Caching**: Aggressive caching (1 year) for immutable Unity assets
- **Security**: Proper content-type headers and security headers

## Deployment

### Prerequisites

1. Install Wrangler CLI:
   ```bash
   npm install -g wrangler
   ```

2. Login to Cloudflare:
   ```bash
   wrangler login
   ```

### Deploy to Production

```bash
cd infrastructure/cloudflare/unity-cdn-worker
wrangler deploy
```

### Deploy to Staging

```bash
wrangler deploy --env staging
```

## Custom Domain Setup

After deployment, set up a custom domain in the Cloudflare dashboard:

1. Go to Workers & Pages > unity-cdn > Settings > Triggers
2. Add Custom Domain: `cdn.curiouskelly.com`
3. Update `unity-kelly-loader.js` to use the new URL

## R2 Bucket Structure

The worker expects files in the R2 bucket with this structure:

```
curious-kelly-unity/
├── Kelly_Web_Build.loader.js
├── Kelly_Web_Build.data.br
├── Kelly_Web_Build.framework.js.br
├── Kelly_Web_Build.wasm.br
└── v1/                          # Optional versioned directory
    ├── Kelly_Web_Build.loader.js
    └── ...
```

## Testing

Test the worker locally:

```bash
wrangler dev
```

Then visit: http://localhost:8787/Kelly_Web_Build.loader.js

## Updating Unity Assets

To upload new Unity builds to R2:

```bash
# From project root
wrangler r2 object put curious-kelly-unity/Kelly_Web_Build.loader.js --file ./path/to/Kelly_Web_Build.loader.js
wrangler r2 object put curious-kelly-unity/Kelly_Web_Build.data.br --file ./path/to/Kelly_Web_Build.data.br
wrangler r2 object put curious-kelly-unity/Kelly_Web_Build.framework.js.br --file ./path/to/Kelly_Web_Build.framework.js.br
wrangler r2 object put curious-kelly-unity/Kelly_Web_Build.wasm.br --file ./path/to/Kelly_Web_Build.wasm.br
```




