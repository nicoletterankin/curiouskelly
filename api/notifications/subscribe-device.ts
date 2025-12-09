/**
 * Subscribe Device to Push Notifications
 * 
 * POST /api/notifications/subscribe-device
 * 
 * Registers a device token for push notifications (iOS, Android, Web).
 * Creates notification preferences if they don't exist.
 * 
 * Body:
 * - device_token: string (required) - APNs token, FCM token, or Web Push subscription
 * - platform: 'ios' | 'android' | 'web' | 'macos' | 'windows' | 'linux'
 * - device_name?: string - User-friendly device name
 * - device_model?: string - Device model
 * - app_version?: string - App version
 * - os_version?: string - OS version
 * 
 * Returns:
 * - success: boolean
 * - token_id: string - ID of the registered token
 * - preferences: object - User's notification preferences
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

interface SubscribeDeviceRequest {
  device_token: string;
  platform: 'ios' | 'android' | 'web' | 'macos' | 'windows' | 'linux';
  device_name?: string;
  device_model?: string;
  app_version?: string;
  os_version?: string;
}

interface NotificationPreferences {
  preferred_time: string;
  timezone: string;
  auto_timing: boolean;
  push_enabled: boolean;
  email_enabled: boolean;
  daily_reminder: boolean;
  streak_alerts: boolean;
  milestone_celebrations: boolean;
  gentle_returns: boolean;
  quiet_start: string;
  quiet_end: string;
  streak_shields_available: number;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Get auth token from header
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Unauthorized - missing auth token' });
  }
  const accessToken = authHeader.replace('Bearer ', '');

  // Initialize Supabase
  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseAnonKey = process.env.PUBLIC_SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseAnonKey) {
    return res.status(500).json({ error: 'Supabase not configured' });
  }

  const supabase = createClient(supabaseUrl, supabaseAnonKey, {
    global: { headers: { Authorization: `Bearer ${accessToken}` } }
  });

  try {
    // Validate request body
    const body = req.body as SubscribeDeviceRequest;
    
    if (!body.device_token) {
      return res.status(400).json({ error: 'device_token is required' });
    }
    
    if (!body.platform || !['ios', 'android', 'web', 'macos', 'windows', 'linux'].includes(body.platform)) {
      return res.status(400).json({ error: 'platform must be one of: ios, android, web, macos, windows, linux' });
    }

    // Get the authenticated user
    const { data: { user }, error: userError } = await supabase.auth.getUser();
    
    if (userError || !user) {
      return res.status(401).json({ error: 'Invalid authentication token' });
    }

    // Upsert the push token
    const { data: token, error: tokenError } = await supabase
      .from('push_tokens')
      .upsert({
        user_id: user.id,
        device_token: body.device_token,
        platform: body.platform,
        device_name: body.device_name || null,
        device_model: body.device_model || null,
        app_version: body.app_version || null,
        os_version: body.os_version || null,
        is_active: true,
        last_active_at: new Date().toISOString(),
        failed_count: 0,
        updated_at: new Date().toISOString()
      }, {
        onConflict: 'user_id,device_token'
      })
      .select('id')
      .single();

    if (tokenError) {
      console.error('Error upserting push token:', tokenError);
      return res.status(500).json({ error: 'Failed to register device', details: tokenError.message });
    }

    // Ensure notification preferences exist for this user
    const { data: existingPrefs, error: prefsCheckError } = await supabase
      .from('notification_preferences')
      .select('*')
      .eq('user_id', user.id)
      .single();

    let preferences: NotificationPreferences;

    if (prefsCheckError || !existingPrefs) {
      // Create default preferences
      const { data: newPrefs, error: createPrefsError } = await supabase
        .from('notification_preferences')
        .insert({
          user_id: user.id,
          preferred_time: '09:00',
          timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'America/New_York',
          auto_timing: true,
          push_enabled: true,
          email_enabled: true,
          web_push_enabled: true,
          daily_reminder: true,
          streak_alerts: true,
          milestone_celebrations: true,
          gentle_returns: true,
          quiet_start: '22:00',
          quiet_end: '07:00',
          weekend_quiet: false,
          streak_shields_available: 0,
          streak_shields_used: 0
        })
        .select()
        .single();

      if (createPrefsError) {
        console.error('Error creating notification preferences:', createPrefsError);
        // Don't fail the request, just log and continue with defaults
        preferences = {
          preferred_time: '09:00',
          timezone: 'America/New_York',
          auto_timing: true,
          push_enabled: true,
          email_enabled: true,
          daily_reminder: true,
          streak_alerts: true,
          milestone_celebrations: true,
          gentle_returns: true,
          quiet_start: '22:00',
          quiet_end: '07:00',
          streak_shields_available: 0
        };
      } else {
        preferences = newPrefs as NotificationPreferences;
      }
    } else {
      preferences = existingPrefs as NotificationPreferences;
    }

    // Log this device registration event
    console.log(`[Notifications] Device registered: ${body.platform} for user ${user.id.slice(0, 8)}...`);

    return res.status(200).json({
      success: true,
      message: 'Device registered for push notifications',
      token_id: token?.id,
      preferences: {
        preferred_time: preferences.preferred_time,
        timezone: preferences.timezone,
        auto_timing: preferences.auto_timing,
        push_enabled: preferences.push_enabled,
        email_enabled: preferences.email_enabled,
        daily_reminder: preferences.daily_reminder,
        streak_alerts: preferences.streak_alerts,
        milestone_celebrations: preferences.milestone_celebrations,
        gentle_returns: preferences.gentle_returns,
        quiet_start: preferences.quiet_start,
        quiet_end: preferences.quiet_end,
        streak_shields_available: preferences.streak_shields_available
      }
    });

  } catch (error) {
    console.error('Error in subscribe-device:', error);
    return res.status(500).json({
      error: 'Failed to register device',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

