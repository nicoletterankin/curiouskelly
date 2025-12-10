// Curious Kelly Service Worker
// Handles push notifications and caching

const CACHE_NAME = 'curious-kelly-v1';
const urlsToCache = [
    '/',
    '/kelly.html',
    '/learn.html',
    '/curriculum.html',
    '/css/brand-colors.css',
    '/images/kelly/kelly-hero.jpeg'
];

// Install event - cache essential assets
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('[SW] Caching essential assets');
                return cache.addAll(urlsToCache);
            })
            .catch((error) => {
                console.log('[SW] Cache failed:', error);
            })
    );
    self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== CACHE_NAME) {
                        console.log('[SW] Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
    self.clients.claim();
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request)
            .then((response) => {
                // Return cached version or fetch from network
                return response || fetch(event.request);
            })
    );
});

// Push notification event
self.addEventListener('push', (event) => {
    console.log('[SW] Push received');
    
    let data = {
        title: "✨ Kelly's going live!",
        body: "Today's lesson is starting. Join millions learning together.",
        icon: '/images/kelly/kelly-icon.png',
        badge: '/images/kelly/kelly-badge.png',
        url: '/kelly.html'
    };
    
    if (event.data) {
        try {
            data = event.data.json();
        } catch (e) {
            data.body = event.data.text();
        }
    }
    
    const options = {
        body: data.body,
        icon: data.icon || '/images/kelly/kelly-icon.png',
        badge: data.badge || '/images/kelly/kelly-badge.png',
        vibrate: [100, 50, 100],
        data: {
            url: data.url || '/kelly.html',
            dateOfArrival: Date.now()
        },
        actions: [
            { action: 'join', title: 'Join Class', icon: '/images/icons/play.png' },
            { action: 'later', title: 'Remind Later', icon: '/images/icons/clock.png' }
        ],
        tag: 'kelly-class-notification',
        renotify: true,
        requireInteraction: true
    };
    
    event.waitUntil(
        self.registration.showNotification(data.title, options)
    );
});

// Notification click event
self.addEventListener('notificationclick', (event) => {
    console.log('[SW] Notification clicked');
    
    event.notification.close();
    
    if (event.action === 'join') {
        // Open the class immediately
        event.waitUntil(
            clients.openWindow(event.notification.data.url || '/kelly.html')
        );
    } else if (event.action === 'later') {
        // Schedule a reminder for 5 minutes later
        console.log('[SW] User requested reminder');
        // In production, this would schedule a delayed notification
    } else {
        // Default: open the class
        event.waitUntil(
            clients.matchAll({ type: 'window' }).then((windowClients) => {
                // Check if there's already a window open
                for (const client of windowClients) {
                    if (client.url.includes('kelly') && 'focus' in client) {
                        return client.focus();
                    }
                }
                // Otherwise open a new window
                if (clients.openWindow) {
                    return clients.openWindow(event.notification.data.url || '/kelly.html');
                }
            })
        );
    }
});

// Background sync for offline lesson tracking
self.addEventListener('sync', (event) => {
    if (event.tag === 'sync-lesson-progress') {
        event.waitUntil(syncLessonProgress());
    }
});

async function syncLessonProgress() {
    // Sync any offline lesson progress to the server
    console.log('[SW] Syncing lesson progress...');
}

console.log('[SW] Service Worker loaded - Curious Kelly v1');










