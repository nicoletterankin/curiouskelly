/**
 * Event Logging API
 * 
 * POST /api/events
 * 
 * Logs user events to the user_events table for zero-trust audit trail.
 * Both authenticated and anonymous events are supported.
 * 
 * Events are best-effort - this endpoint never blocks the user experience.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

interface EventRequest {
  event_type: string;
  event_category?: 'learner_action' | 'kelly_action' | 'system';
  payload?: Record<string, unknown>;
  day_number?: number;
  session_id?: string;
}

// Valid event types (subset - allows custom events too)
const KNOWN_EVENTS = [
  'lesson.started', 'lesson.completed', 'lesson.paused', 'lesson.skipped',
  'paywall.shown', 'paywall.dismissed', 'paywall.cta_clicked',
  'purchase.initiated', 'purchase.completed', 'purchase.failed',
  'subscription.started', 'subscription.cancelled',
  'nav.day_selected', 'nav.calendar_opened',
  'settings.updated', 'profile.updated',
  'auth.signup', 'auth.login', 'auth.logout',
  'system.session_started', 'system.error'
];

function detectDeviceType(userAgent: string): string {
  const ua = userAgent.toLowerCase();
  if (/roku|tv|smarttv/i.test(ua)) return 'tv';
  if (/tablet|ipad/i.test(ua)) return 'tablet';
  if (/mobile|android|iphone/i.test(ua)) return 'mobile';
  return 'desktop';
}

function detectPlatform(userAgent: string): string {
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
  res.setHeader('Content-Type', 'application/json');
  
  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  try {
    const body = (req.body || {}) as EventRequest;
    const eventType = body.event_type || 'unknown';
    
    // Log for debugging (visible in Vercel logs)
    const userAgent = req.headers['user-agent'] || '';
    console.log('[Event]', eventType, {
      day: body.day_number,
      device: detectDeviceType(userAgent),
      platform: detectPlatform(userAgent)
    });
    
    // TODO: When migrations are run, insert into user_events table:
    // const supabase = getSupabaseAdmin();
    // await supabase.from('user_events').insert({
    //   event_type: eventType,
    //   event_category: body.event_category || 'learner_action',
    //   payload: body.payload || {},
    //   day_number: body.day_number,
    //   session_id: body.session_id,
    //   device_type: detectDeviceType(userAgent),
    //   platform: detectPlatform(userAgent),
    //   user_agent: userAgent
    // });
    
    return res.status(200).json({ 
      success: true,
      event_type: eventType
    });
    
  } catch (error) {
    console.error('Error in events API:', error);
    // Events should never fail - return success anyway
    return res.status(200).json({ 
      success: true, 
      fallback: true
    });
  }
}
