/**
 * Test Push Notification Endpoint
 * 
 * POST /api/notifications/test-push
 * 
 * Sends a test push notification to a specific device or user.
 * Used for verifying push notification setup works correctly.
 * 
 * Body:
 * - user_id: string (optional) - Send to all user's devices
 * - token_id: string (optional) - Send to specific device token
 * - title: string (optional) - Custom title
 * - body: string (optional) - Custom body
 * 
 * Returns:
 * - success: boolean
 * - results: Array of send results
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';
import { sendToUser, sendPushNotification, type PushPayload, type PushToken } from '../../lib/push-sender';
import { getTodayLessonDay, getLessonDateStrings } from '../../lib/lesson-dates';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Require admin authorization in production
  const adminSecret = process.env.ADMIN_SECRET || process.env.CRON_SECRET;
  const authHeader = req.headers.authorization;
  
  if (adminSecret && authHeader !== `Bearer ${adminSecret}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!supabaseUrl || !supabaseServiceKey) {
    return res.status(500).json({ error: 'Supabase not configured' });
  }

  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  try {
    const { user_id, token_id, title, body } = req.body;

    if (!user_id && !token_id) {
      return res.status(400).json({ 
        error: 'Either user_id or token_id is required' 
      });
    }

    // Get today's info for default message
    const todayDay = getTodayLessonDay();
    const dateInfo = getLessonDateStrings(todayDay);

    // Construct test payload
    const payload: PushPayload = {
      title: title || '🧪 Test from Kelly!',
      body: body || `This is a test notification for ${dateInfo.formatted}. If you see this, push is working!`,
      url: `https://curiouskelly.com/day/${todayDay}`,
      tag: 'kelly-test',
      data: {
        test: true,
        timestamp: new Date().toISOString()
      }
    };

    if (token_id) {
      // Send to specific token
      const { data: token, error } = await supabase
        .from('push_tokens')
        .select('*')
        .eq('id', token_id)
        .single();

      if (error || !token) {
        return res.status(404).json({ error: 'Token not found' });
      }

      const result = await sendPushNotification(token as PushToken, payload, supabase);

      // Log the test
      await supabase.from('notification_log').insert({
        user_id: token.user_id,
        notification_type: 'test',
        title: payload.title,
        body: payload.body,
        platform: token.platform,
        device_token_id: token.id,
        sent_at: new Date().toISOString(),
        error_message: result.error || null,
        metadata: { test: true }
      });

      return res.status(200).json({
        success: result.success,
        platform: token.platform,
        result
      });
    }

    if (user_id) {
      // Send to all user's devices
      const results = await sendToUser(user_id, payload, supabase);

      // Log the test
      for (const result of results.results) {
        await supabase.from('notification_log').insert({
          user_id,
          notification_type: 'test',
          title: payload.title,
          body: payload.body,
          platform: result.platform,
          sent_at: new Date().toISOString(),
          error_message: result.error || null,
          metadata: { test: true }
        });
      }

      return res.status(200).json({
        success: results.sent > 0,
        message: `Sent to ${results.sent} devices, ${results.failed} failed`,
        stats: {
          sent: results.sent,
          failed: results.failed
        },
        results: results.results
      });
    }

  } catch (error) {
    console.error('[Test Push] Error:', error);
    return res.status(500).json({
      error: 'Failed to send test notification',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

