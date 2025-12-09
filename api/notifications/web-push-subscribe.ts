/**
 * Web Push Subscription Endpoint (Public)
 * 
 * POST /api/notifications/web-push-subscribe
 * 
 * Registers a web push subscription. Does NOT require authentication.
 * For anonymous users, creates an anonymous token that can later be
 * associated with a user account.
 * 
 * Body:
 * - endpoint: string (required) - Web push endpoint URL
 * - p256dh: string (required) - Web push p256dh key
 * - auth: string (required) - Web push auth key
 * - device_id: string (optional) - Browser fingerprint
 * - timezone: string (optional) - User's timezone
 * - user_id: string (optional) - If user is logged in
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers for browser requests
  res.setHeader('Access-Control-Allow-Origin', 'https://curiouskelly.com');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Use service role for unauthenticated access
  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!supabaseUrl || !supabaseServiceKey) {
    return res.status(500).json({ error: 'Supabase not configured' });
  }

  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  try {
    const { endpoint, p256dh, auth, device_id, timezone, user_id } = req.body;

    // Validate required fields
    if (!endpoint) {
      return res.status(400).json({ error: 'endpoint is required' });
    }
    if (!p256dh) {
      return res.status(400).json({ error: 'p256dh key is required' });
    }
    if (!auth) {
      return res.status(400).json({ error: 'auth key is required' });
    }

    // Store subscription as JSON (format expected by web-push library)
    const device_token = JSON.stringify({ endpoint, p256dh, auth });

    // Check if this endpoint already exists
    const { data: existing } = await supabase
      .from('push_tokens')
      .select('id, user_id')
      .eq('device_token', device_token)
      .single();

    let tokenId: string;

    if (existing) {
      // Update existing token
      const updateData: Record<string, unknown> = {
        is_active: true,
        last_active_at: new Date().toISOString(),
        failed_count: 0,
        updated_at: new Date().toISOString()
      };

      // Only update user_id if provided and different
      if (user_id && user_id !== existing.user_id) {
        updateData.user_id = user_id;
      }

      const { error: updateError } = await supabase
        .from('push_tokens')
        .update(updateData)
        .eq('id', existing.id);

      if (updateError) throw updateError;
      tokenId = existing.id;

      console.log(`[Web Push] Updated subscription for ${user_id || 'anonymous'}`);
    } else {
      // Insert new token
      const { data: newToken, error: insertError } = await supabase
        .from('push_tokens')
        .insert({
          user_id: user_id || null,
          device_token,
          platform: 'web',
          device_name: device_id ? `Browser ${device_id.slice(-6)}` : 'Web Browser',
          is_active: true,
          last_active_at: new Date().toISOString(),
          failed_count: 0
        })
        .select('id')
        .single();

      if (insertError) throw insertError;
      tokenId = newToken.id;

      console.log(`[Web Push] Created new subscription for ${user_id || 'anonymous'}`);
    }

    // If user_id provided and timezone available, update notification preferences
    if (user_id && timezone) {
      await supabase
        .from('notification_preferences')
        .upsert({
          user_id,
          timezone,
          updated_at: new Date().toISOString()
        }, {
          onConflict: 'user_id'
        });
    }

    return res.status(200).json({
      success: true,
      message: 'Web push subscription registered',
      token_id: tokenId
    });

  } catch (error) {
    console.error('[Web Push] Error:', error);
    return res.status(500).json({
      error: 'Failed to register subscription',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

