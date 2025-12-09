/**
 * Curious Kelly Push Notification System
 * Enables daily class reminders and engagement notifications
 */

const PushNotifications = {
    // VAPID public key for web push (private key stored in Vercel env vars)
    // Generated: December 2025 for curiouskelly.com
    VAPID_PUBLIC_KEY: 'BEgmu91QD3hye9UZ9MM6xZxfbIRrmhiKE3cV3XkfvxAlRMATdRY4skdaFAMKVyKNkZJmXKGW2otkUEFcqUqnsOg',
    
    // API endpoint for storing subscriptions (public, no auth required)
    SUBSCRIPTION_ENDPOINT: '/api/notifications/web-push-subscribe',
    
    /**
     * Check if push notifications are supported
     */
    isSupported() {
        return 'serviceWorker' in navigator && 'PushManager' in window;
    },
    
    /**
     * Request permission for push notifications
     */
    async requestPermission() {
        if (!this.isSupported()) {
            console.log('[Push] Not supported in this browser');
            return false;
        }
        
        const permission = await Notification.requestPermission();
        console.log('[Push] Permission:', permission);
        return permission === 'granted';
    },
    
    /**
     * Register the service worker
     */
    async registerServiceWorker() {
        try {
            const registration = await navigator.serviceWorker.register('/sw.js');
            console.log('[Push] Service Worker registered:', registration.scope);
            return registration;
        } catch (error) {
            console.error('[Push] Service Worker registration failed:', error);
            return null;
        }
    },
    
    /**
     * Subscribe to push notifications
     */
    async subscribe() {
        if (!await this.requestPermission()) {
            return null;
        }
        
        const registration = await this.registerServiceWorker();
        if (!registration) return null;
        
        try {
            // Wait for the service worker to be ready
            await navigator.serviceWorker.ready;
            
            // Subscribe to push
            const subscription = await registration.pushManager.subscribe({
                userVisibleOnly: true,
                applicationServerKey: this.urlBase64ToUint8Array(this.VAPID_PUBLIC_KEY)
            });
            
            console.log('[Push] Subscribed:', subscription);
            
            // Store the subscription on the server
            await this.saveSubscription(subscription);
            
            return subscription;
        } catch (error) {
            console.error('[Push] Subscription failed:', error);
            return null;
        }
    },
    
    /**
     * Unsubscribe from push notifications
     */
    async unsubscribe() {
        const registration = await navigator.serviceWorker.ready;
        const subscription = await registration.pushManager.getSubscription();
        
        if (subscription) {
            await subscription.unsubscribe();
            console.log('[Push] Unsubscribed');
            return true;
        }
        
        return false;
    },
    
    /**
     * Check current subscription status
     */
    async getSubscription() {
        if (!this.isSupported()) return null;
        
        const registration = await navigator.serviceWorker.ready;
        return await registration.pushManager.getSubscription();
    },
    
    /**
     * Save subscription to server
     */
    async saveSubscription(subscription) {
        const subscriptionJson = subscription.toJSON();
        const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
        
        // Store locally as backup
        localStorage.setItem('kelly-push-subscription', JSON.stringify({
            subscription: subscriptionJson,
            timezone: timezone,
            createdAt: new Date().toISOString()
        }));
        
        // Send to Supabase via API
        try {
            const response = await fetch(this.SUBSCRIPTION_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    endpoint: subscriptionJson.endpoint,
                    p256dh: subscriptionJson.keys.p256dh,
                    auth: subscriptionJson.keys.auth,
                    platform: 'web',
                    device_id: this.getDeviceId(),
                    timezone: timezone
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const result = await response.json();
            console.log('[Push] Subscription saved to server:', result);
            return result;
        } catch (error) {
            console.error('[Push] Failed to save subscription to server:', error);
            // Subscription still works locally, will sync on next visit
            return null;
        }
    },
    
    /**
     * Get or create a unique device ID for this browser
     */
    getDeviceId() {
        let deviceId = localStorage.getItem('kelly-device-id');
        if (!deviceId) {
            deviceId = 'web_' + crypto.randomUUID();
            localStorage.setItem('kelly-device-id', deviceId);
        }
        return deviceId;
    },
    
    /**
     * Convert VAPID key for web push
     */
    urlBase64ToUint8Array(base64String) {
        const padding = '='.repeat((4 - base64String.length % 4) % 4);
        const base64 = (base64String + padding)
            .replace(/-/g, '+')
            .replace(/_/g, '/');
        
        const rawData = window.atob(base64);
        const outputArray = new Uint8Array(rawData.length);
        
        for (let i = 0; i < rawData.length; ++i) {
            outputArray[i] = rawData.charCodeAt(i);
        }
        
        return outputArray;
    },
    
    /**
     * Schedule a local notification (for testing)
     */
    async scheduleLocalNotification(title, body, delay = 5000) {
        if (Notification.permission !== 'granted') {
            await this.requestPermission();
        }
        
        setTimeout(() => {
            new Notification(title, {
                body: body,
                icon: '/images/kelly/kelly-icon.png',
                badge: '/images/kelly/kelly-badge.png',
                tag: 'kelly-local-test'
            });
        }, delay);
    },
    
    /**
     * Create the notification prompt UI
     */
    createPromptUI() {
        // Don't show if already subscribed
        if (localStorage.getItem('kelly-push-subscription')) {
            return null;
        }
        
        const prompt = document.createElement('div');
        prompt.id = 'push-notification-prompt';
        prompt.innerHTML = `
            <div class="push-prompt">
                <div class="push-prompt-content">
                    <div class="push-prompt-icon">🔔</div>
                    <div class="push-prompt-text">
                        <h4>Never miss a class with Kelly!</h4>
                        <p>Get a gentle reminder when today's lesson starts.</p>
                    </div>
                </div>
                <div class="push-prompt-actions">
                    <button class="push-prompt-btn push-prompt-later">Maybe later</button>
                    <button class="push-prompt-btn push-prompt-enable">Enable notifications</button>
                </div>
            </div>
        `;
        
        // Add styles
        const styles = document.createElement('style');
        styles.textContent = `
            #push-notification-prompt {
                position: fixed;
                bottom: 24px;
                right: 24px;
                z-index: 10000;
                animation: slideUp 0.3s ease-out;
            }
            
            @keyframes slideUp {
                from { transform: translateY(100px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
            
            .push-prompt {
                background: #1c1c24;
                border: 1px solid rgba(37, 99, 235, 0.3);
                border-radius: 16px;
                padding: 20px;
                max-width: 360px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            }
            
            .push-prompt-content {
                display: flex;
                gap: 12px;
                margin-bottom: 16px;
            }
            
            .push-prompt-icon {
                font-size: 32px;
            }
            
            .push-prompt-text h4 {
                color: #f4f4f5;
                font-size: 1rem;
                margin: 0 0 4px 0;
            }
            
            .push-prompt-text p {
                color: #a1a1aa;
                font-size: 0.9rem;
                margin: 0;
            }
            
            .push-prompt-actions {
                display: flex;
                gap: 12px;
            }
            
            .push-prompt-btn {
                flex: 1;
                padding: 10px 16px;
                border-radius: 8px;
                font-size: 0.9rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
                border: none;
            }
            
            .push-prompt-later {
                background: transparent;
                color: #a1a1aa;
            }
            
            .push-prompt-later:hover {
                color: #f4f4f5;
            }
            
            .push-prompt-enable {
                background: #2563eb;
                color: white;
            }
            
            .push-prompt-enable:hover {
                background: #3b82f6;
            }
            
            @media (max-width: 480px) {
                #push-notification-prompt {
                    left: 12px;
                    right: 12px;
                    bottom: 12px;
                }
                
                .push-prompt {
                    max-width: none;
                }
            }
        `;
        
        document.head.appendChild(styles);
        document.body.appendChild(prompt);
        
        // Handle actions
        prompt.querySelector('.push-prompt-later').addEventListener('click', () => {
            prompt.remove();
            // Show again after 24 hours
            localStorage.setItem('kelly-push-prompt-dismissed', Date.now());
        });
        
        prompt.querySelector('.push-prompt-enable').addEventListener('click', async () => {
            const subscription = await this.subscribe();
            if (subscription) {
                prompt.innerHTML = `
                    <div class="push-prompt">
                        <div class="push-prompt-content">
                            <div class="push-prompt-icon">✅</div>
                            <div class="push-prompt-text">
                                <h4>You're all set!</h4>
                                <p>We'll notify you when Kelly goes live.</p>
                            </div>
                        </div>
                    </div>
                `;
                setTimeout(() => prompt.remove(), 3000);
            }
        });
        
        return prompt;
    },
    
    /**
     * Initialize push notifications with smart prompting
     */
    async init() {
        if (!this.isSupported()) {
            console.log('[Push] Not supported');
            return;
        }
        
        // Register service worker early
        await this.registerServiceWorker();
        
        // Check if already subscribed
        const existing = await this.getSubscription();
        if (existing) {
            console.log('[Push] Already subscribed');
            return;
        }
        
        // Check if user dismissed recently
        const dismissed = localStorage.getItem('kelly-push-prompt-dismissed');
        if (dismissed && Date.now() - parseInt(dismissed) < 24 * 60 * 60 * 1000) {
            console.log('[Push] User dismissed recently, waiting...');
            return;
        }
        
        // Show prompt after 30 seconds of engagement
        setTimeout(() => {
            this.createPromptUI();
        }, 30000);
    }
};

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => PushNotifications.init());
} else {
    PushNotifications.init();
}

// Export for use in other scripts
window.PushNotifications = PushNotifications;








