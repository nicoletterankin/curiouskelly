// Curious Kelly Service Worker
// App-shell caching + (optional) push notifications

const CACHE_NAME = 'curious-kelly-app-v4';

const APP_SHELL = [
  '/learn.html',
  '/manifest.json',
  '/sw.js',
  '/styles/kelly-foundation.css',
  '/js/kelly-time.js',
  '/js/kelly-calendar.js',
  '/js/kelly-lesson.js',
  '/js/kelly-presence.js',
  '/assets/kelly/kelly-personas-manifest.json',
  '/images/brand/android-chrome-192.png',
  '/images/brand/android-chrome-512.png',
  '/images/brand/apple-touch-icon.png'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(APP_SHELL))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);

  // Network-first for API, Supabase, and runtime config (avoid stale keys)
  if (
    url.pathname.startsWith('/api/') ||
    url.hostname.includes('supabase') ||
    url.pathname === '/config.js' ||
    url.pathname === '/sw.js' ||
    url.pathname.endsWith('.html')
  ) {
    event.respondWith(
      fetch(req).catch(() => caches.match(req))
    );
    return;
  }

  // Cache-first for same-origin static
  if (url.origin === self.location.origin) {
    event.respondWith(
      caches.match(req).then((cached) => cached || fetch(req).then((res) => {
        const copy = res.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(req, copy)).catch(() => {});
        return res;
      }))
    );
  }
});

// Push notification event (copy must be honest and non-deceptive)
self.addEventListener('push', (event) => {
  let data = {
    title: "✨ Today's lesson is ready",
    body: "Open Curious Kelly when you're ready to learn.",
    icon: '/images/brand/android-chrome-192.png',
    badge: '/images/brand/android-chrome-192.png',
    url: '/learn.html'
  };

  if (event.data) {
    try {
      data = { ...data, ...event.data.json() };
    } catch (e) {
      data.body = event.data.text();
    }
  }

  const options = {
    body: data.body,
    icon: data.icon,
    badge: data.badge,
    data: { url: data.url, dateOfArrival: Date.now() },
    tag: 'kelly-daily-lesson',
    renotify: true
  };

  event.waitUntil(self.registration.showNotification(data.title, options));
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then((windowClients) => {
      for (const client of windowClients) {
        if (client.url.includes('/learn') && 'focus' in client) return client.focus();
      }
      return clients.openWindow(event.notification.data?.url || '/learn.html');
    })
  );
});










