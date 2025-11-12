import path from 'node:path';
import { generateSW } from 'workbox-build';

const distDir = path.resolve(process.cwd(), 'dist');
const swDest = path.join(distDir, 'service-worker.js');

async function main() {
  try {
    const { count, size, warnings } = await generateSW({
      globDirectory: distDir,
      globPatterns: ['**/*.{html,js,css,svg,png,webp,avif,ico,json,txt,xml}'],
      swDest,
      sourcemap: false,
      maximumFileSizeToCacheInBytes: 3 * 1024 * 1024,
      clientsClaim: true,
      skipWaiting: false,
      cleanupOutdatedCaches: true,
      navigateFallback: '/thank-you/index.html',
      runtimeCaching: [
        {
          urlPattern: ({ request }) => request.destination === 'image',
          handler: 'CacheFirst',
          options: {
            cacheName: 'images',
            expiration: {
              maxEntries: 60,
              maxAgeSeconds: 60 * 60 * 24 * 30
            },
            matchOptions: {
              ignoreSearch: true
            }
          }
        },
        {
          urlPattern: ({ url }) => url.pathname.startsWith('/api/lead') || url.pathname.startsWith('/api/rum'),
          handler: 'NetworkOnly'
        },
        {
          urlPattern: ({ request }) => request.destination === 'document',
          handler: 'NetworkFirst',
          options: {
            cacheName: 'pages',
            networkTimeoutSeconds: 3
          }
        }
      ]
    });

    warnings.forEach((warning) => console.warn('[workbox]', warning));
    console.log(`[workbox] Generated ${swDest} with ${count} precached file(s), total ${size} bytes`);
  } catch (error) {
    console.error('[workbox] Failed to build service worker', error);
    process.exitCode = 1;
  }
}

main();




