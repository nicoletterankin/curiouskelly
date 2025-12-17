/**
 * Kelly Resilience Layer
 * 
 * Graceful degradation when Supabase or other services are unavailable.
 * THE LESSON ALWAYS PLAYS.
 * 
 * Principles:
 * 1. Anonymous users can always take lessons (localStorage progress)
 * 2. Auth failures don't block lesson playback
 * 3. Payment checks fail OPEN for free content
 * 4. Progress is saved locally, synced when possible
 */

(function() {
  'use strict';

  const STORAGE_PREFIX = 'kelly_';
  
  // ═══════════════════════════════════════════════════════════════════
  // SERVICE STATUS TRACKING
  // ═══════════════════════════════════════════════════════════════════
  
  const ServiceStatus = {
    supabase: 'unknown',
    stripe: 'unknown',
    api: 'unknown',
    lastCheck: null,
    
    check: async function() {
      this.lastCheck = Date.now();
      
      // Check API health
      try {
        const response = await fetch('/api/health', { 
          method: 'GET',
          timeout: 5000 
        });
        const data = await response.json();
        this.api = 'healthy';
        this.supabase = data.checks?.database?.status === 'ok' ? 'healthy' : 'degraded';
      } catch (e) {
        this.api = 'down';
        this.supabase = 'unknown';
      }
      
      return this;
    },
    
    isSupabaseHealthy: function() {
      return this.supabase === 'healthy';
    }
  };
  
  // ═══════════════════════════════════════════════════════════════════
  // LOCAL PROGRESS (Works Offline)
  // ═══════════════════════════════════════════════════════════════════
  
  const LocalProgress = {
    getCompletedDays: function() {
      try {
        const stored = localStorage.getItem(STORAGE_PREFIX + 'completed_days');
        return stored ? JSON.parse(stored) : [];
      } catch (e) {
        return [];
      }
    },
    
    markDayComplete: function(dayNumber) {
      try {
        const days = this.getCompletedDays();
        if (!days.includes(dayNumber)) {
          days.push(dayNumber);
          localStorage.setItem(STORAGE_PREFIX + 'completed_days', JSON.stringify(days));
        }
        this.updateStreak();
        return true;
      } catch (e) {
        console.warn('[Resilience] Could not save progress:', e);
        return false;
      }
    },
    
    getStreak: function() {
      try {
        return parseInt(localStorage.getItem(STORAGE_PREFIX + 'streak') || '0');
      } catch (e) {
        return 0;
      }
    },
    
    updateStreak: function() {
      try {
        const lastDate = localStorage.getItem(STORAGE_PREFIX + 'last_lesson_date');
        const today = new Date().toISOString().split('T')[0];
        const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];
        
        let streak = this.getStreak();
        
        if (lastDate === today) {
          // Already counted today
        } else if (lastDate === yesterday) {
          // Streak continues
          streak++;
        } else if (!lastDate) {
          // First lesson ever
          streak = 1;
        } else {
          // Streak broken
          streak = 1;
        }
        
        localStorage.setItem(STORAGE_PREFIX + 'streak', streak.toString());
        localStorage.setItem(STORAGE_PREFIX + 'last_lesson_date', today);
        
        return streak;
      } catch (e) {
        return 1;
      }
    },
    
    getTotalLessons: function() {
      return this.getCompletedDays().length;
    },
    
    getPreferences: function() {
      try {
        const stored = localStorage.getItem(STORAGE_PREFIX + 'preferences');
        return stored ? JSON.parse(stored) : {
          age: 30,
          language: 'en',
          archetype: 'The Explorer'
        };
      } catch (e) {
        return { age: 30, language: 'en', archetype: 'The Explorer' };
      }
    },
    
    setPreferences: function(prefs) {
      try {
        const current = this.getPreferences();
        const merged = { ...current, ...prefs };
        localStorage.setItem(STORAGE_PREFIX + 'preferences', JSON.stringify(merged));
        return true;
      } catch (e) {
        return false;
      }
    }
  };
  
  // ═══════════════════════════════════════════════════════════════════
  // AUTH RESILIENCE (Guest Mode)
  // ═══════════════════════════════════════════════════════════════════
  
  const AuthResilience = {
    /**
     * Get current user (from Supabase or guest mode)
     * NEVER throws - returns guest user if auth unavailable
     */
    getCurrentUser: async function() {
      // Try Supabase first
      try {
        const supabase = window.getSupabase?.();
        if (supabase) {
          const { data: { user }, error } = await supabase.auth.getUser();
          if (user && !error) {
            return {
              id: user.id,
              email: user.email,
              name: user.user_metadata?.full_name || user.email?.split('@')[0],
              isGuest: false,
              source: 'supabase'
            };
          }
        }
      } catch (e) {
        console.warn('[Resilience] Auth check failed, using guest mode:', e.message);
      }
      
      // Fallback: Guest user
      return this.getGuestUser();
    },
    
    getGuestUser: function() {
      let guestId = localStorage.getItem(STORAGE_PREFIX + 'guest_id');
      if (!guestId) {
        guestId = 'guest_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem(STORAGE_PREFIX + 'guest_id', guestId);
      }
      
      return {
        id: guestId,
        email: null,
        name: 'Guest Learner',
        isGuest: true,
        source: 'local'
      };
    },
    
    /**
     * Check if user can access a lesson
     * FAILS OPEN for Day 1-7 (trial) + any completed days
     */
    canAccessLesson: async function(dayNumber) {
      // First 7 days are always accessible (trial)
      if (dayNumber <= 7) {
        return { allowed: true, reason: 'trial' };
      }
      
      // Already completed days are accessible (re-watch)
      const completed = LocalProgress.getCompletedDays();
      if (completed.includes(dayNumber)) {
        return { allowed: true, reason: 'completed' };
      }
      
      // Try to check subscription status
      try {
        const user = await this.getCurrentUser();
        if (user.isGuest) {
          return { allowed: false, reason: 'guest_limit', message: 'Sign up for full access' };
        }
        
        // Check with API (with timeout)
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 3000);
        
        const response = await fetch('/api/subscription-status', {
          signal: controller.signal,
          headers: {
            'Content-Type': 'application/json'
          }
        });
        clearTimeout(timeout);
        
        const data = await response.json();
        if (data.isSubscribed || data.hasLifetime) {
          return { allowed: true, reason: 'subscribed' };
        }
      } catch (e) {
        // If we can't check subscription, FAIL OPEN for existing users
        console.warn('[Resilience] Subscription check failed, allowing access:', e.message);
        return { allowed: true, reason: 'offline_grace' };
      }
      
      return { allowed: false, reason: 'not_subscribed' };
    }
  };
  
  // ═══════════════════════════════════════════════════════════════════
  // LESSON COMPLETION (Resilient)
  // ═══════════════════════════════════════════════════════════════════
  
  const LessonCompletion = {
    /**
     * Mark a lesson complete
     * Saves locally FIRST, then tries to sync with server
     */
    complete: async function(dayNumber) {
      // Always save locally first
      LocalProgress.markDayComplete(dayNumber);
      
      // Try to sync with server (non-blocking)
      this.syncWithServer(dayNumber).catch(e => {
        console.warn('[Resilience] Server sync failed, will retry later:', e.message);
        this.queueForSync(dayNumber);
      });
      
      return {
        success: true,
        streak: LocalProgress.getStreak(),
        totalLessons: LocalProgress.getTotalLessons()
      };
    },
    
    syncWithServer: async function(dayNumber) {
      try {
        const user = await AuthResilience.getCurrentUser();
        if (user.isGuest) {
          return; // No server sync for guests
        }
        
        await fetch('/api/lesson-complete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dayNumber,
            completedAt: new Date().toISOString()
          })
        });
        
        // Clear from sync queue if successful
        this.removeFromQueue(dayNumber);
      } catch (e) {
        throw e;
      }
    },
    
    queueForSync: function(dayNumber) {
      try {
        const queue = JSON.parse(localStorage.getItem(STORAGE_PREFIX + 'sync_queue') || '[]');
        if (!queue.includes(dayNumber)) {
          queue.push(dayNumber);
          localStorage.setItem(STORAGE_PREFIX + 'sync_queue', JSON.stringify(queue));
        }
      } catch (e) {
        // Ignore
      }
    },
    
    removeFromQueue: function(dayNumber) {
      try {
        const queue = JSON.parse(localStorage.getItem(STORAGE_PREFIX + 'sync_queue') || '[]');
        const filtered = queue.filter(d => d !== dayNumber);
        localStorage.setItem(STORAGE_PREFIX + 'sync_queue', JSON.stringify(filtered));
      } catch (e) {
        // Ignore
      }
    },
    
    processSyncQueue: async function() {
      try {
        const queue = JSON.parse(localStorage.getItem(STORAGE_PREFIX + 'sync_queue') || '[]');
        for (const dayNumber of queue) {
          await this.syncWithServer(dayNumber);
        }
      } catch (e) {
        console.warn('[Resilience] Queue sync failed:', e.message);
      }
    }
  };
  
  // ═══════════════════════════════════════════════════════════════════
  // GLOBAL API
  // ═══════════════════════════════════════════════════════════════════
  
  window.KellyResilience = {
    ServiceStatus,
    LocalProgress,
    AuthResilience,
    LessonCompletion,
    
    // Convenience methods
    getProgress: () => ({
      streak: LocalProgress.getStreak(),
      totalLessons: LocalProgress.getTotalLessons(),
      completedDays: LocalProgress.getCompletedDays()
    }),
    
    getCurrentUser: () => AuthResilience.getCurrentUser(),
    canAccess: (day) => AuthResilience.canAccessLesson(day),
    completeLesson: (day) => LessonCompletion.complete(day),
    
    // Health check
    checkHealth: () => ServiceStatus.check()
  };
  
  // Auto-process sync queue when online
  if (typeof window !== 'undefined') {
    window.addEventListener('online', () => {
      console.log('[Resilience] Back online, processing sync queue...');
      LessonCompletion.processSyncQueue();
    });
    
    // Check health on load
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => ServiceStatus.check());
    } else {
      ServiceStatus.check();
    }
  }
  
  console.log('✅ Kelly Resilience Layer loaded - THE LESSON ALWAYS PLAYS');
  
})();
