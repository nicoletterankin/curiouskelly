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
import { getSupabaseAdmin, isSupabaseConfigured } from './lib/supabase';

interface EventRequest {
  event_type: string;
  event_category?: 'learner_action' | 'kelly_action' | 'system';
  payload?: Record<string, unknown>;
  day_number?: number;
  session_id?: string;
}

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

function getClientIP(req: VercelRequest): string | null {
  const forwarded = req.headers['x-forwarded-for'];
  if (typeof forwarded === 'string') {
    return forwarded.split(',')[0].trim();
  }
  return null;
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
    const userAgent = req.headers['user-agent'] || '';
    
    // Log for debugging (visible in Vercel logs)
    console.log('[Event]', eventType, {
      day: body.day_number,
      device: detectDeviceType(userAgent),
      platform: detectPlatform(userAgent)
    });
    
    // Try to store in database if configured
    let stored = false;
    if (isSupabaseConfigured()) {
      try {
        const supabase = getSupabaseAdmin();
        
        // Extract user ID from auth header if present
        let userId: string | null = null;
        const authHeader = req.headers['authorization'];
        if (authHeader && authHeader.startsWith('Bearer ')) {
          const token = authHeader.substring(7);
          try {
            const { data: { user } } = await supabase.auth.getUser(token);
            userId = user?.id || null;
          } catch {
            // Token invalid, continue without user ID
          }
        }
        
        const { error } = await supabase.from('user_events').insert({
          user_id: userId,
          event_type: eventType,
          event_category: body.event_category || 'learner_action',
          payload: body.payload || {},
          day_number: body.day_number,
          session_id: body.session_id,
          device_type: detectDeviceType(userAgent),
          platform: detectPlatform(userAgent),
          user_agent: userAgent.substring(0, 500), // Limit length
          ip_address: getClientIP(req)
        });
        
        if (!error) {
          stored = true;
        } else if (error.code === '42P01') {
          // Table doesn't exist yet - migrations not run
          console.log('[Event] user_events table not yet created');
        } else {
          console.error('[Event] Insert error:', error.message);
        }
      } catch (dbError) {
        console.error('[Event] Database error:', dbError);
      }
    }
    
    return res.status(200).json({ 
      success: true,
      event_type: eventType,
      stored
    });
    
  } catch (error) {
    console.error('Error in events API:', error);
    // Events should never fail the request - return success anyway
    return res.status(200).json({ 
      success: true, 
      fallback: true
    });
  }
}
