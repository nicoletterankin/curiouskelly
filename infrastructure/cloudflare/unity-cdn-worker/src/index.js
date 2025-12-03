/**
 * Unity CDN Worker
 * 
 * Serves Unity WebGL build assets from Cloudflare R2 with:
 * - Proper CORS headers for cross-origin requests
 * - Content-Encoding headers for Brotli-compressed files
 * - Aggressive caching for immutable assets
 * - Security headers
 * 
 * @see https://developers.cloudflare.com/r2/api/workers/workers-api-usage/
 */

// Allowed origins for CORS
const ALLOWED_ORIGINS = [
  'https://curiouskelly.com',
  'https://www.curiouskelly.com',
  'https://learn.curiouskelly.com',
  'http://localhost:3000',
  'http://localhost:4321',
  'http://127.0.0.1:3000',
  'http://127.0.0.1:4321',
];

// MIME types for Unity WebGL files
const MIME_TYPES = {
  '.data': 'application/octet-stream',
  '.data.br': 'application/octet-stream',
  '.wasm': 'application/wasm',
  '.wasm.br': 'application/wasm',
  '.js': 'application/javascript',
  '.js.br': 'application/javascript',
  '.json': 'application/json',
  '.json.br': 'application/json',
};

// Cache durations
const CACHE_CONTROL = {
  versioned: 'public, max-age=31536000, immutable', // 1 year for versioned files
  default: 'public, max-age=86400', // 1 day for other files
};

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;

    // Handle CORS preflight requests
    if (request.method === 'OPTIONS') {
      return handleCorsPrelight(request);
    }

    // Only allow GET and HEAD requests
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return new Response('Method Not Allowed', { status: 405 });
    }

    // Remove leading slash for R2 key
    const key = path.startsWith('/') ? path.slice(1) : path;

    // Handle root path
    if (!key || key === '/') {
      return new Response('Unity CDN - Asset server for Curious Kelly', {
        status: 200,
        headers: getCorsHeaders(request),
      });
    }

    try {
      // Fetch from R2
      const object = await env.UNITY_BUCKET.get(key);

      if (!object) {
        // Try with version prefix if not found
        // e.g., /Kelly_Web_Build.loader.js -> /v1/Kelly_Web_Build.loader.js
        const versionedKey = `v1/${key}`;
        const versionedObject = await env.UNITY_BUCKET.get(versionedKey);
        
        if (!versionedObject) {
          return new Response('Asset not found', { 
            status: 404,
            headers: getCorsHeaders(request),
          });
        }
        
        return buildResponse(request, versionedObject, versionedKey);
      }

      return buildResponse(request, object, key);
    } catch (error) {
      console.error('Error fetching from R2:', error);
      return new Response(`Error: ${error.message}`, { 
        status: 500,
        headers: getCorsHeaders(request),
      });
    }
  },
};

/**
 * Build the response with proper headers
 */
function buildResponse(request, object, key) {
  const headers = new Headers();
  
  // CORS headers
  const corsHeaders = getCorsHeaders(request);
  for (const [name, value] of Object.entries(corsHeaders)) {
    headers.set(name, value);
  }

  // Content-Type
  const contentType = getContentType(key);
  headers.set('Content-Type', contentType);

  // Content-Encoding for Brotli files
  if (key.endsWith('.br')) {
    headers.set('Content-Encoding', 'br');
  } else if (key.endsWith('.gz')) {
    headers.set('Content-Encoding', 'gzip');
  }

  // Cache-Control - aggressive caching for Unity assets
  // Unity WebGL files are versioned by build, so they're effectively immutable
  const isVersioned = key.includes('v1/') || key.includes('v2/') || /\.[a-f0-9]{8}\./.test(key);
  headers.set('Cache-Control', isVersioned ? CACHE_CONTROL.versioned : CACHE_CONTROL.versioned);

  // ETag from R2
  if (object.httpEtag) {
    headers.set('ETag', object.httpEtag);
  }

  // Content-Length
  headers.set('Content-Length', object.size);

  // Security headers
  headers.set('X-Content-Type-Options', 'nosniff');

  // Handle conditional requests
  const ifNoneMatch = request.headers.get('If-None-Match');
  if (ifNoneMatch && object.httpEtag === ifNoneMatch) {
    return new Response(null, { status: 304, headers });
  }

  return new Response(object.body, { status: 200, headers });
}

/**
 * Get content type based on file extension
 */
function getContentType(key) {
  // Remove .br or .gz suffix for base extension
  const baseKey = key.replace(/\.(br|gz)$/, '');
  
  for (const [ext, type] of Object.entries(MIME_TYPES)) {
    if (key.endsWith(ext)) {
      return type;
    }
  }

  // Check base extension
  if (baseKey.endsWith('.js')) return 'application/javascript';
  if (baseKey.endsWith('.wasm')) return 'application/wasm';
  if (baseKey.endsWith('.data')) return 'application/octet-stream';
  if (baseKey.endsWith('.json')) return 'application/json';
  
  return 'application/octet-stream';
}

/**
 * Handle CORS preflight requests
 */
function handleCorsPrelight(request) {
  const headers = getCorsHeaders(request);
  headers['Access-Control-Max-Age'] = '86400'; // 24 hours
  
  return new Response(null, {
    status: 204,
    headers,
  });
}

/**
 * Get CORS headers based on request origin
 */
function getCorsHeaders(request) {
  const origin = request.headers.get('Origin');
  
  // Check if origin is allowed
  const isAllowed = !origin || ALLOWED_ORIGINS.some(allowed => {
    if (allowed.includes('localhost') || allowed.includes('127.0.0.1')) {
      // Match any localhost port
      return origin.startsWith('http://localhost:') || origin.startsWith('http://127.0.0.1:');
    }
    return origin === allowed || origin.endsWith(allowed.replace('https://', '.'));
  });

  // For development, allow all origins; in production, be stricter
  const allowOrigin = isAllowed ? (origin || '*') : ALLOWED_ORIGINS[0];

  return {
    'Access-Control-Allow-Origin': allowOrigin,
    'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Range, Accept-Encoding',
    'Access-Control-Expose-Headers': 'Content-Length, Content-Range, Content-Encoding, ETag',
    'Access-Control-Max-Age': '86400',
  };
}

