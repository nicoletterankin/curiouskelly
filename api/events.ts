/**
 * Event Logging API
 * 
 * POST /api/events
 * 
 * Logs user events to the user_events table for zero-trust audit trail.
 * Both authenticated and anonymous events are supported.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getSupabaseAdmin, isSupabaseConfigured } from './lib/supabase';

interface EventRequest {
  event_type: string;
  event_category?: 'learner_action' | 'kelly_action' | 'system';
  payload?: Record<string, unknown>;
  day_number?: number;
  session_id?: string;
}

// Valid event types for learner actions
const VALID_LEARNER_EVENTS = [
  // Lesson events
  'lesson.started', 'lesson.completed', 'lesson.paused', 'lesson.resumed', 
  'lesson.skipped', 'lesson.replayed', 'lesson.question_answered',
  // Navigation
  'nav.day_selected', 'nav.calendar_opened', 'nav.search',
  // Comments
  'comment.posted', 'comment.edited', 'comment.deleted', 'comment.reported',
  // Artwork
  'artwork.submitted', 'artwork.withdrawn',
  // Reactions
  'reaction.added', 'reaction.removed',
  // Purchases
  'purchase.initiated', 'purchase.completed', 'purchase.failed', 'purchase.refunded',
  // Subscription
  'subscription.started', 'subscription.renewed', 'subscription.cancelled', 'subscription.paused',
  // Live classes
  'liveclass.joined', 'liveclass.left', 'liveclass.question',
  // Downloads
  'download.requested', 'download.completed', 'download.bundle',
  // Settings
  'settings.updated', 'profile.updated', 'preferences.updated',
  // Auth
  'auth.signup', 'auth.login', 'auth.logout',
  // Support
  'support.ticket_opened', 'support.message_sent',
  // Paywall
  'paywall.shown', 'paywall.dismissed', 'paywall.cta_clicked'
];

// Valid event types for Kelly actions (server-side only)
const VALID_KELLY_EVENTS = [
  'kelly.email_sent', 'kelly.push_sent', 'kelly.sms_sent',
  'kelly.reminder_sent', 'kelly.streak_celebrated',
  'kelly.welcome_sent', 'kelly.comeback_sent',
  'kelly.gift_delivered', 'kelly.birthday_message',
  'moderation.comment_approved', 'moderation.comment_rejected',
  'moderation.artwork_approved', 'moderation.artwork_rejected'
];

const VALID_SYSTEM_EVENTS = [
  'system.session_started', 'system.session_ended',
  'system.error', 'system.migration', 'system.health_check'
];

function detectDeviceType(userAgent: string): string {
  const ua = userAgent.toLowerCase();
  if (/roku|tv|smarttv|television/i.test(ua)) return 'tv';
  if (/tablet|ipad/i.test(ua)) return 'tablet';
  if (/mobile|android|iphone/i.test(ua)) return 'mobile';
  return 'desktop';
}

function detectPlatform(userAgent: string, referer?: string): string {
  const ua = userAgent.toLowerCase();
  if (/roku/i.test(ua)) return 'roku';
  if (/android/i.test(ua) && /wv/i.test(ua)) return 'android';
  if (/iphone|ipad/i.test(ua) && !/safari/i.test(ua)) return 'ios';
  return 'web';
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  if (!isSupabaseConfigured()) {
    // Still return success - events are best-effort
    return res.status(200).json({ 
      success: true, 
      warning: 'Database not configured',
      fallback: true
    });
  }
  
  let supabase;
  try {
    supabase = getSupabaseAdmin();
  } catch (e) {
    return res.status(200).json({ 
      success: true, 
      warning: 'Database init failed',
      fallback: true
    });
  }
  
  try {
    const body = req.body as EventRequest;
    
    // Validate event_type
    if (!body.event_type) {
      return res.status(400).json({ error: 'event_type is required' });
    }
    
    // Determine category and validate
    let category = body.event_category;
    
    if (!category) {
      // Auto-detect category
      if (VALID_LEARNER_EVENTS.includes(body.event_type)) {
        category = 'learner_action';
      } else if (VALID_KELLY_EVENTS.includes(body.event_type)) {
        category = 'kelly_action';
      } else if (VALID_SYSTEM_EVENTS.includes(body.event_type)) {
        category = 'system';
      } else {
        // Allow custom events but default to learner_action
        category = 'learner_action';
      }
    }
    
    // Get user from auth token if provided
    let userId: string | null = null;
    const authHeader = req.headers.authorization;
    
    if (authHeader?.startsWith('Bearer ')) {
      const token = authHeader.substring(7);
      const { data: { user } } = await supabase.auth.getUser(token);
      if (user) {
        userId = user.id;
      }
    }
    
    // Get request metadata
    const userAgent = req.headers['user-agent'] || '';
    const ip = (req.headers['x-forwarded-for'] as string)?.split(',')[0]?.trim() 
      || req.headers['x-real-ip'] as string 
      || req.socket?.remoteAddress 
      || null;
    
    // Build event record
    const eventRecord = {
      user_id: userId,
      session_id: body.session_id || null,
      event_type: body.event_type,
      event_category: category,
      payload: body.payload || {},
      day_number: body.day_number || null,
      ip_address: ip,
      user_agent: userAgent,
      device_type: detectDeviceType(userAgent),
      platform: detectPlatform(userAgent, req.headers.referer as string)
    };
    
    // Try to insert event - but gracefully fail if table doesn't exist yet
    try {
      const { error: insertError } = await supabase
        .from('user_events')
        .insert(eventRecord);
      
      if (insertError) {
        console.error('Event insert error:', insertError);
        // Don't fail the request - events are best-effort
        // Table might not exist yet (migration not run)
        return res.status(200).json({ 
          success: true, 
          warning: 'Event logged (table pending)',
          fallback: true
        });
      }
    } catch (dbError) {
      // Table might not exist - that's OK for now
      console.warn('Event logging failed (table may not exist):', dbError);
      return res.status(200).json({ 
        success: true, 
        warning: 'Event logging pending (run migrations)',
        fallback: true
      });
    }
    
    return res.status(200).json({ 
      success: true,
      event_type: body.event_type
    });
    
  } catch (error) {
    console.error('Error in events API:', error);
    // Events should never block the user experience
    return res.status(200).json({ 
      success: true, 
      warning: 'Event logging failed silently',
      fallback: true
    });
  }
}
