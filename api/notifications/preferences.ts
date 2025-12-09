/**
 * Notification Preferences API
 * 
 * GET /api/notifications/preferences - Get user's notification preferences
 * PUT /api/notifications/preferences - Update user's notification preferences
 * 
 * Handles all user notification settings including:
 * - Timing (preferred time, timezone, auto-timing)
 * - Channels (push, email, web)
 * - Types (daily reminder, streak alerts, milestones, etc.)
 * - Quiet hours
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

interface NotificationPreferences {
  preferred_time: string;
  timezone: string;
  auto_timing: boolean;
  learned_optimal_time: string | null;
  push_enabled: boolean;
  email_enabled: boolean;
  web_push_enabled: boolean;
  daily_reminder: boolean;
  streak_alerts: boolean;
  milestone_celebrations: boolean;
  gentle_returns: boolean;
  family_updates: boolean;
  collective_milestones: boolean;
  quiet_start: string;
  quiet_end: string;
  weekend_quiet: boolean;
  streak_shields_available: number;
  streak_shields_used: number;
}

interface UpdatePreferencesRequest {
  preferred_time?: string;
  timezone?: string;
  auto_timing?: boolean;
  push_enabled?: boolean;
  email_enabled?: boolean;
  web_push_enabled?: boolean;
  daily_reminder?: boolean;
  streak_alerts?: boolean;
  milestone_celebrations?: boolean;
  gentle_returns?: boolean;
  family_updates?: boolean;
  collective_milestones?: boolean;
  quiet_start?: string;
  quiet_end?: string;
  weekend_quiet?: boolean;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Only allow GET and PUT
  if (!['GET', 'PUT'].includes(req.method || '')) {
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
    // Get the authenticated user
    const { data: { user }, error: userError } = await supabase.auth.getUser();
    
    if (userError || !user) {
      return res.status(401).json({ error: 'Invalid authentication token' });
    }

    if (req.method === 'GET') {
      // GET: Retrieve preferences
      const { data: preferences, error: prefsError } = await supabase
        .from('notification_preferences')
        .select('*')
        .eq('user_id', user.id)
        .single();

      if (prefsError) {
        // If not found, return defaults
        if (prefsError.code === 'PGRST116') {
          return res.status(200).json({
            success: true,
            preferences: {
              preferred_time: '09:00',
              timezone: 'America/New_York',
              auto_timing: true,
              learned_optimal_time: null,
              push_enabled: true,
              email_enabled: true,
              web_push_enabled: true,
              daily_reminder: true,
              streak_alerts: true,
              milestone_celebrations: true,
              gentle_returns: true,
              family_updates: false,
              collective_milestones: false,
              quiet_start: '22:00',
              quiet_end: '07:00',
              weekend_quiet: false,
              streak_shields_available: 0,
              streak_shields_used: 0
            },
            is_default: true
          });
        }
        throw prefsError;
      }

      return res.status(200).json({
        success: true,
        preferences: {
          preferred_time: preferences.preferred_time,
          timezone: preferences.timezone,
          auto_timing: preferences.auto_timing,
          learned_optimal_time: preferences.learned_optimal_time,
          push_enabled: preferences.push_enabled,
          email_enabled: preferences.email_enabled,
          web_push_enabled: preferences.web_push_enabled,
          daily_reminder: preferences.daily_reminder,
          streak_alerts: preferences.streak_alerts,
          milestone_celebrations: preferences.milestone_celebrations,
          gentle_returns: preferences.gentle_returns,
          family_updates: preferences.family_updates,
          collective_milestones: preferences.collective_milestones,
          quiet_start: preferences.quiet_start,
          quiet_end: preferences.quiet_end,
          weekend_quiet: preferences.weekend_quiet,
          streak_shields_available: preferences.streak_shields_available,
          streak_shields_used: preferences.streak_shields_used
        },
        is_default: false
      });

    } else {
      // PUT: Update preferences
      const body = req.body as UpdatePreferencesRequest;

      // Validate time formats if provided
      const timeRegex = /^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/;
      
      if (body.preferred_time && !timeRegex.test(body.preferred_time)) {
        return res.status(400).json({ error: 'Invalid preferred_time format. Use HH:MM' });
      }
      if (body.quiet_start && !timeRegex.test(body.quiet_start)) {
        return res.status(400).json({ error: 'Invalid quiet_start format. Use HH:MM' });
      }
      if (body.quiet_end && !timeRegex.test(body.quiet_end)) {
        return res.status(400).json({ error: 'Invalid quiet_end format. Use HH:MM' });
      }

      // Build update object with only provided fields
      const updateData: Record<string, unknown> = {
        updated_at: new Date().toISOString()
      };

      const allowedFields = [
        'preferred_time', 'timezone', 'auto_timing',
        'push_enabled', 'email_enabled', 'web_push_enabled',
        'daily_reminder', 'streak_alerts', 'milestone_celebrations',
        'gentle_returns', 'family_updates', 'collective_milestones',
        'quiet_start', 'quiet_end', 'weekend_quiet'
      ];

      for (const field of allowedFields) {
        if (body[field as keyof UpdatePreferencesRequest] !== undefined) {
          updateData[field] = body[field as keyof UpdatePreferencesRequest];
        }
      }

      // If user turns off auto_timing, clear the learned time
      if (body.auto_timing === false) {
        updateData.learned_optimal_time = null;
      }

      // Upsert to handle case where preferences don't exist yet
      const { data: updatedPrefs, error: updateError } = await supabase
        .from('notification_preferences')
        .upsert({
          user_id: user.id,
          ...updateData
        }, {
          onConflict: 'user_id'
        })
        .select()
        .single();

      if (updateError) {
        console.error('Error updating notification preferences:', updateError);
        return res.status(500).json({ error: 'Failed to update preferences', details: updateError.message });
      }

      // Also update user's email preferences in users table if relevant
      if (body.email_enabled !== undefined || body.daily_reminder !== undefined) {
        const userUpdate: Record<string, boolean> = {};
        
        if (body.email_enabled !== undefined && body.daily_reminder !== undefined) {
          userUpdate.email_daily_lesson = body.email_enabled && body.daily_reminder;
        } else if (body.email_enabled !== undefined) {
          userUpdate.email_daily_lesson = body.email_enabled;
        } else if (body.daily_reminder !== undefined) {
          userUpdate.email_daily_lesson = body.daily_reminder;
        }

        if (Object.keys(userUpdate).length > 0) {
          await supabase
            .from('users')
            .update(userUpdate)
            .eq('id', user.id);
        }
      }

      console.log(`[Notifications] Preferences updated for user ${user.id.slice(0, 8)}...`);

      return res.status(200).json({
        success: true,
        message: 'Preferences updated successfully',
        preferences: {
          preferred_time: updatedPrefs.preferred_time,
          timezone: updatedPrefs.timezone,
          auto_timing: updatedPrefs.auto_timing,
          learned_optimal_time: updatedPrefs.learned_optimal_time,
          push_enabled: updatedPrefs.push_enabled,
          email_enabled: updatedPrefs.email_enabled,
          web_push_enabled: updatedPrefs.web_push_enabled,
          daily_reminder: updatedPrefs.daily_reminder,
          streak_alerts: updatedPrefs.streak_alerts,
          milestone_celebrations: updatedPrefs.milestone_celebrations,
          gentle_returns: updatedPrefs.gentle_returns,
          family_updates: updatedPrefs.family_updates,
          collective_milestones: updatedPrefs.collective_milestones,
          quiet_start: updatedPrefs.quiet_start,
          quiet_end: updatedPrefs.quiet_end,
          weekend_quiet: updatedPrefs.weekend_quiet,
          streak_shields_available: updatedPrefs.streak_shields_available,
          streak_shields_used: updatedPrefs.streak_shields_used
        }
      });
    }

  } catch (error) {
    console.error('Error in notification preferences:', error);
    return res.status(500).json({
      error: 'Failed to process preferences request',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}


