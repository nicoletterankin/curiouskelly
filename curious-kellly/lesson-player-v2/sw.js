/**
 * Kelly Service Worker
 * 
 * Provides offline support for the Curious Kelly lesson player.
 * Caches lessons, audio, and images for offline learning.
 * 
 * Features:
 * - Offline lesson access
 * - Background sync for completion tracking
 * - Push notifications for daily reminders
 * - Efficient cache management
 * 
 * @version 1.0.0
 * @lastUpdated December 2025
 */

const CACHE_VERSION = 'kelly-v1.0.0';
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const DYNAMIC_CACHE = `${CACHE_VERSION}-dynamic`;
const LESSON_CACHE = `${CACHE_VERSION}-lessons`;
const AUDIO_CACHE = `${CACHE_VERSION}-audio`;

// Static assets to pre-cache
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/css/styles.css',
  '/css/ui-kit.css',
  '/js/app.js',
  '/js/kelly-2d-avatar.js',
  '/js/kelly-unified-avatar.js',
  '/js/kelly-lipsync-player.js',
  '/js/kelly-voice-engine.js',
  '/js/kelly-age-adaptive-avatar.js',
  '/js/kelly-accessibility.js',
  '/js/kelly-settings.js',
  '/js/kelly-touch.js',
  '/js/lesson-history.js',
  '/js/earn-to-learn.js'
];

// Maximum cache sizes
const MAX_DYNAMIC_CACHE = 50;
const MAX_LESSON_CACHE = 30;  // 30 days of lessons
const MAX_AUDIO_CACHE = 100;

// =============================================================================
// INSTALL EVENT
// =============================================================================

self.addEventListener('install', (event) => {
  console.log('[SW] Installing service worker...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then(cache => {
        console.log('[SW] Pre-caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => self.skipWaiting())
  );
});

// =============================================================================
// ACTIVATE EVENT
// =============================================================================

self.addEventListener('activate', (event) => {
  console.log('[SW] Activating service worker...');
  
  event.waitUntil(
    caches.keys()
      .then(keys => {
        return Promise.all(
          keys
            .filter(key => key.startsWith('kelly-') && key !== CACHE_VERSION)
            .map(key => {
              console.log('[SW] Deleting old cache:', key);
              return caches.delete(key);
            })
        );
      })
      .then(() => self.clients.claim())
  );
});

// =============================================================================
// FETCH EVENT
// =============================================================================

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') return;
  
  // Skip external requests
  if (!url.origin.includes(self.location.origin)) return;
  
  // Route to appropriate cache strategy
  if (isStaticAsset(url)) {
    event.respondWith(cacheFirst(request, STATIC_CACHE));
  } else if (isLessonAsset(url)) {
    event.respondWith(cacheFirst(request, LESSON_CACHE));
  } else if (isAudioAsset(url)) {
    event.respondWith(cacheFirst(request, AUDIO_CACHE));
  } else if (isAPIRequest(url)) {
    event.respondWith(networkFirst(request));
  } else {
    event.respondWith(staleWhileRevalidate(request, DYNAMIC_CACHE));
  }
});

// =============================================================================
// CACHE STRATEGIES
// =============================================================================

/**
 * Cache first, then network
 */
async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  
  if (cached) {
    return cached;
  }
  
  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    return offlineFallback(request);
  }
}

/**
 * Network first, then cache
 */
async function networkFirst(request) {
  try {
    const response = await fetch(request);
    return response;
  } catch (error) {
    const cached = await caches.match(request);
    if (cached) {
      return cached;
    }
    return offlineFallback(request);
  }
}

/**
 * Stale while revalidate
 */
async function staleWhileRevalidate(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  
  const networkPromise = fetch(request)
    .then(response => {
      if (response.ok) {
        cache.put(request, response.clone());
        trimCache(cacheName, MAX_DYNAMIC_CACHE);
      }
      return response;
    })
    .catch(() => cached || offlineFallback(request));
  
  return cached || networkPromise;
}

/**
 * Offline fallback
 */
function offlineFallback(request) {
  const url = new URL(request.url);
  
  // Return offline page for HTML requests
  if (request.headers.get('accept')?.includes('text/html')) {
    return caches.match('/index.html');
  }
  
  // Return placeholder for images
  if (isImageAsset(url)) {
    return new Response(OFFLINE_IMAGE_SVG, {
      headers: { 'Content-Type': 'image/svg+xml' }
    });
  }
  
  // Return error for other requests
  return new Response('Offline', { status: 503, statusText: 'Offline' });
}

// =============================================================================
// ASSET DETECTION
// =============================================================================

function isStaticAsset(url) {
  return STATIC_ASSETS.some(asset => url.pathname.endsWith(asset)) ||
    url.pathname.endsWith('.css') ||
    url.pathname.endsWith('.js') && !url.pathname.includes('/api/');
}

function isLessonAsset(url) {
  return url.pathname.includes('/lessons/') && url.pathname.endsWith('.json');
}

function isAudioAsset(url) {
  return url.pathname.endsWith('.mp3') ||
    url.pathname.endsWith('.wav') ||
    url.pathname.endsWith('.ogg');
}

function isImageAsset(url) {
  return url.pathname.endsWith('.png') ||
    url.pathname.endsWith('.jpg') ||
    url.pathname.endsWith('.jpeg') ||
    url.pathname.endsWith('.webp') ||
    url.pathname.endsWith('.svg');
}

function isAPIRequest(url) {
  return url.pathname.startsWith('/api/');
}

// =============================================================================
// CACHE MANAGEMENT
// =============================================================================

async function trimCache(cacheName, maxItems) {
  const cache = await caches.open(cacheName);
  const keys = await cache.keys();
  
  if (keys.length > maxItems) {
    // Delete oldest entries
    const toDelete = keys.slice(0, keys.length - maxItems);
    await Promise.all(toDelete.map(key => cache.delete(key)));
  }
}

// =============================================================================
// BACKGROUND SYNC
// =============================================================================

self.addEventListener('sync', (event) => {
  console.log('[SW] Background sync:', event.tag);
  
  if (event.tag === 'sync-lesson-completion') {
    event.waitUntil(syncLessonCompletions());
  }
});

async function syncLessonCompletions() {
  // Get pending completions from IndexedDB
  // POST to /api/lesson-complete
  console.log('[SW] Syncing lesson completions...');
}

// =============================================================================
// PUSH NOTIFICATIONS
// =============================================================================

self.addEventListener('push', (event) => {
  const data = event.data?.json() || {};
  
  const options = {
    body: data.body || 'Time for today\'s lesson!',
    icon: '/images/kelly-icon-192.png',
    badge: '/images/kelly-badge.png',
    vibrate: [100, 50, 100],
    data: {
      url: data.url || '/'
    },
    actions: [
      { action: 'open', title: 'Learn Now' },
      { action: 'later', title: 'Remind Later' }
    ]
  };
  
  event.waitUntil(
    self.registration.showNotification(data.title || '✨ Curious Kelly', options)
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  if (event.action === 'later') {
    // Schedule another notification in 1 hour
    return;
  }
  
  event.waitUntil(
    clients.matchAll({ type: 'window' })
      .then(windowClients => {
        // Focus existing window or open new
        for (const client of windowClients) {
          if (client.url === event.notification.data.url && 'focus' in client) {
            return client.focus();
          }
        }
        return clients.openWindow(event.notification.data.url);
      })
  );
});

// =============================================================================
// MESSAGES
// =============================================================================

self.addEventListener('message', (event) => {
  const { action, data } = event.data || {};
  
  switch (action) {
    case 'CACHE_LESSON':
      cacheLessonForOffline(data.dayNumber);
      break;
      
    case 'CLEAR_CACHE':
      clearAllCaches();
      break;
      
    case 'GET_CACHE_STATUS':
      getCacheStatus().then(status => {
        event.ports[0].postMessage(status);
      });
      break;
  }
});

async function cacheLessonForOffline(dayNumber) {
  const paddedDay = String(dayNumber).padStart(2, '0');
  const lessonPath = `/lessons/daily/day-${paddedDay}`;
  
  const assets = [
    `${lessonPath}/lesson.json`,
    `${lessonPath}/audio/hook.mp3`,
    `${lessonPath}/audio/q1.mp3`,
    `${lessonPath}/audio/q2.mp3`,
    `${lessonPath}/audio/q3.mp3`,
    `${lessonPath}/audio/wisdom.mp3`,
    `${lessonPath}/images/hook.png`,
    `${lessonPath}/images/q1.png`,
    `${lessonPath}/images/q2.png`,
    `${lessonPath}/images/q3.png`,
    `${lessonPath}/images/wisdom.png`
  ];
  
  const cache = await caches.open(LESSON_CACHE);
  await cache.addAll(assets);
  console.log(`[SW] Cached lesson ${dayNumber} for offline`);
}

async function clearAllCaches() {
  const keys = await caches.keys();
  await Promise.all(keys.map(key => caches.delete(key)));
  console.log('[SW] All caches cleared');
}

async function getCacheStatus() {
  const cacheNames = [STATIC_CACHE, DYNAMIC_CACHE, LESSON_CACHE, AUDIO_CACHE];
  const status = {};
  
  for (const name of cacheNames) {
    const cache = await caches.open(name);
    const keys = await cache.keys();
    status[name] = keys.length;
  }
  
  return status;
}

// =============================================================================
// OFFLINE PLACEHOLDER
// =============================================================================

const OFFLINE_IMAGE_SVG = `
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
  <rect fill="#27272a" width="200" height="200"/>
  <text x="100" y="100" text-anchor="middle" fill="#71717a" font-family="system-ui" font-size="14">
    Offline
  </text>
</svg>
`;

console.log('[SW] Service worker loaded');







