/**
 * Daily Push Notification Cron Job
 * 
 * Sends daily lesson reminder push notifications to all platforms:
 * - iOS (APNs)
 * - Android (FCM)
 * - Web (VAPID)
 * 
 * Runs at the top of every hour to catch users in their optimal time.
 * Respects quiet hours, timezone, and user preferences.
 * 
 * Schedule: "0 * * * *" (every hour at :00)
 * 
 * Environment Variables:
 * - CRON_SECRET: Secret for cron authorization
 * - FIREBASE_PROJECT_ID: Firebase project for FCM
 * - FIREBASE_PRIVATE_KEY: Firebase service account private key
 * - FIREBASE_CLIENT_EMAIL: Firebase service account email
 * - APNS_KEY_ID: Apple APNs key ID
 * - APNS_TEAM_ID: Apple team ID
 * - APNS_PRIVATE_KEY: APNs private key (.p8 contents)
 * - VAPID_PUBLIC_KEY: Web push public key
 * - VAPID_PRIVATE_KEY: Web push private key
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';
import { 
  getTodayLessonDay, 
  getLessonDateStrings, 
  getSpecialDateInfo,
  getLessonUrl 
} from '../../lib/lesson-dates';
import {
  sendWebPush as webPushSend,
  sendAPNs as apnsSend,
  sendFCM as fcmSend,
  type PushPayload
} from '../../lib/push-sender';

// Types
interface UserWithPrefs {
  id: string;
  display_name: string | null;
  name: string | null;
  current_streak: number | null;
  push_tokens: PushToken[];
  notification_preferences: NotificationPreference | null;
}

interface PushToken {
  id: string;
  device_token: string;
  platform: 'ios' | 'android' | 'web' | 'macos' | 'windows' | 'linux';
  is_active: boolean;
}

interface NotificationPreference {
  preferred_time: string;
  timezone: string;
  auto_timing: boolean;
  learned_optimal_time: string | null;
  push_enabled: boolean;
  daily_reminder: boolean;
  quiet_start: string;
  quiet_end: string;
  weekend_quiet: boolean;
}

interface Lesson {
  day_number: number;
  title: string;
  emoji: string;
}

interface NotificationCopy {
  variant_code: string;
  title: string;
  body: string;
}

// Note: Day number calculation moved to lib/lesson-dates.ts
// Use getTodayLessonDay() for the internal day number (1-365)
// Use getLessonDateStrings() for user-facing date formatting

// Get current hour in a timezone
function getCurrentHourInTimezone(timezone: string): number {
  try {
    const formatter = new Intl.DateTimeFormat('en-US', {
      timeZone: timezone,
      hour: 'numeric',
      hour12: false
    });
    return parseInt(formatter.format(new Date()), 10);
  } catch {
    return new Date().getHours(); // Fallback to server time
  }
}

// Check if current time is within quiet hours
function isQuietHours(quietStart: string, quietEnd: string, timezone: string): boolean {
  const currentHour = getCurrentHourInTimezone(timezone);
  const startHour = parseInt(quietStart.split(':')[0], 10);
  const endHour = parseInt(quietEnd.split(':')[0], 10);
  
  // Handle overnight quiet hours (e.g., 22:00 - 07:00)
  if (startHour > endHour) {
    return currentHour >= startHour || currentHour < endHour;
  }
  
  return currentHour >= startHour && currentHour < endHour;
}

// Check if user's optimal notification hour matches current hour
function isOptimalHour(preference: NotificationPreference, currentHourUTC: number): boolean {
  const timezone = preference.timezone || 'America/New_York';
  const currentHourLocal = getCurrentHourInTimezone(timezone);
  
  // Get their preferred hour
  const preferredTime = preference.auto_timing && preference.learned_optimal_time
    ? preference.learned_optimal_time
    : preference.preferred_time || '09:00';
  
  const preferredHour = parseInt(preferredTime.split(':')[0], 10);
  
  return currentHourLocal === preferredHour;
}

// Personalize notification copy with real dates (not day numbers)
function personalizeCopy(
  copy: NotificationCopy,
  userName: string,
  lesson: Lesson,
  streak: number
): { title: string; body: string } {
  // Get formatted date strings for user-facing content
  const dateInfo = getLessonDateStrings(lesson.day_number);
  const specialDate = getSpecialDateInfo(lesson.day_number);
  
  const replacements: Record<string, string> = {
    '{name}': userName,
    '{lesson_title}': lesson.title,
    '{lesson_emoji}': lesson.emoji || '📚',
    '{streak_days}': streak.toString(),
    // Date formatting (user-facing) - NO day numbers shown to users
    '{date_formatted}': dateInfo.formatted,           // "December 17"
    '{date_short}': dateInfo.formattedShort,          // "Dec 17"
    '{date_with_year}': dateInfo.formattedWithYear,   // "December 17, 2025"
    '{date_weekday}': dateInfo.formattedWithWeekday,  // "Wednesday, December 17"
    '{day_of_week}': dateInfo.dayOfWeek,              // "Wednesday"
    '{month_name}': dateInfo.monthName,               // "December"
    // Special occasions
    '{special_occasion}': specialDate.specialOccasion || '',
  };
  
  let title = copy.title;
  let body = copy.body;
  
  for (const [key, value] of Object.entries(replacements)) {
    title = title.replace(new RegExp(key, 'g'), value);
    body = body.replace(new RegExp(key, 'g'), value);
  }
  
  return { title, body };
}

// Send iOS push notification (APNs)
async function sendAPNs(token: string, title: string, body: string, dayNumber: number): Promise<boolean> {
  const payload: PushPayload = {
    title,
    body,
    url: getLessonUrl(dayNumber),
    data: { dayNumber: dayNumber.toString() }
  };
  
  const result = await apnsSend(token, payload);
  
  if (!result.success) {
    console.error(`[APNs] Failed to send: ${result.error}`);
  }
  
  return result.success;
}

// Send Android push notification (FCM)
async function sendFCM(token: string, title: string, body: string, dayNumber: number): Promise<boolean> {
  const payload: PushPayload = {
    title,
    body,
    url: getLessonUrl(dayNumber),
    data: { dayNumber: dayNumber.toString() }
  };
  
  const result = await fcmSend(token, payload);
  
  if (!result.success) {
    console.error(`[FCM] Failed to send: ${result.error}`);
  }
  
  return result.success;
}

// Send Web push notification (VAPID)
async function sendWebPush(subscriptionJson: string, title: string, body: string, dayNumber: number): Promise<boolean> {
  try {
    const subscription = JSON.parse(subscriptionJson);
    const payload: PushPayload = {
      title,
      body,
      url: getLessonUrl(dayNumber),
      tag: 'kelly-daily-lesson',
      data: { dayNumber: dayNumber.toString() }
    };
    
    const result = await webPushSend(
      subscription.endpoint,
      subscription.p256dh,
      subscription.auth,
      payload
    );
    
    if (!result.success) {
      console.error(`[WebPush] Failed to send: ${result.error}`);
    }
    
    return result.success;
  } catch (error) {
    console.error('[WebPush] Invalid subscription JSON:', error);
    return false;
  }
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Verify cron secret
  const cronSecret = process.env.CRON_SECRET;
  const authHeader = req.headers.authorization;
  
  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!supabaseUrl || !supabaseServiceKey) {
    return res.status(500).json({ error: 'Supabase not configured' });
  }

  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  try {
    // Get today's lesson day (internal: 1-365, starting Dec 17)
    const todayLessonDay = getTodayLessonDay();
    const dateInfo = getLessonDateStrings(todayLessonDay);
    const currentHourUTC = new Date().getUTCHours();
    const isWeekend = [0, 6].includes(new Date().getDay());
    
    console.log(`[Push Cron] Starting for ${dateInfo.formatted} (day ${todayLessonDay}), hour ${currentHourUTC} UTC`);

    // Get today's lesson
    const { data: lesson } = await supabase
      .from('lessons')
      .select('day_number, title, emoji')
      .eq('day_number', todayLessonDay)
      .single();

    if (!lesson) {
      return res.status(200).json({
        success: true,
        message: 'No lesson found for today',
        stats: { sent: 0 }
      });
    }

    // Get a random copy variant for this hour
    const { data: copyVariants } = await supabase
      .from('notification_copy')
      .select('variant_code, title, body')
      .eq('notification_type', 'daily_reminder')
      .eq('is_active', true);

    if (!copyVariants || copyVariants.length === 0) {
      console.error('[Push Cron] No notification copy found');
      return res.status(500).json({ error: 'No notification copy configured' });
    }

    // Randomly select a variant (in production, use A/B testing logic)
    const selectedCopy = copyVariants[Math.floor(Math.random() * copyVariants.length)];

    // Get users who should receive notifications this hour
    // This includes users with push tokens who:
    // - Have push_enabled = true
    // - Have daily_reminder = true
    // - Are at their optimal notification hour
    // - Are not in quiet hours
    // - Haven't completed today's lesson yet
    const { data: users, error: usersError } = await supabase
      .from('users')
      .select(`
        id,
        display_name,
        name,
        current_streak,
        push_tokens (
          id,
          device_token,
          platform,
          is_active
        ),
        notification_preferences (
          preferred_time,
          timezone,
          auto_timing,
          learned_optimal_time,
          push_enabled,
          daily_reminder,
          quiet_start,
          quiet_end,
          weekend_quiet
        )
      `)
      .not('push_tokens', 'is', null);

    if (usersError) {
      console.error('[Push Cron] Error fetching users:', usersError);
      return res.status(500).json({ error: 'Failed to fetch users' });
    }

    if (!users || users.length === 0) {
      return res.status(200).json({
        success: true,
        message: 'No users with push tokens',
        stats: { sent: 0 }
      });
    }

    // Process users
    let sentCount = 0;
    let skippedCount = 0;
    let failedCount = 0;
    const errors: string[] = [];

    for (const user of users as UserWithPrefs[]) {
      // Skip if no preferences (use defaults)
      const prefs = user.notification_preferences || {
        preferred_time: '09:00',
        timezone: 'America/New_York',
        auto_timing: true,
        learned_optimal_time: null,
        push_enabled: true,
        daily_reminder: true,
        quiet_start: '22:00',
        quiet_end: '07:00',
        weekend_quiet: false
      };

      // Check if push is enabled
      if (!prefs.push_enabled || !prefs.daily_reminder) {
        skippedCount++;
        continue;
      }

      // Check weekend quiet
      if (isWeekend && prefs.weekend_quiet) {
        skippedCount++;
        continue;
      }

      // Check if this is their optimal hour
      if (!isOptimalHour(prefs, currentHourUTC)) {
        skippedCount++;
        continue;
      }

      // Check quiet hours
      if (isQuietHours(prefs.quiet_start, prefs.quiet_end, prefs.timezone)) {
        skippedCount++;
        continue;
      }

      // Check if user already completed today's lesson
      const { count: completedCount } = await supabase
        .from('lesson_completions')
        .select('*', { count: 'exact', head: true })
        .eq('user_id', user.id)
        .eq('day_number', todayLessonDay)
        .gte('completed_at', new Date(new Date().setHours(0, 0, 0, 0)).toISOString());

      if ((completedCount || 0) > 0) {
        skippedCount++;
        continue;
      }

      // Personalize the notification
      const userName = user.display_name || user.name || 'friend';
      const streak = user.current_streak || 0;
      const { title, body } = personalizeCopy(selectedCopy, userName, lesson, streak);

      // Send to all active tokens for this user
      for (const token of user.push_tokens || []) {
        if (!token.is_active) continue;

        let success = false;

        try {
          switch (token.platform) {
            case 'ios':
              success = await sendAPNs(token.device_token, title, body, todayLessonDay);
              break;
            case 'android':
              success = await sendFCM(token.device_token, title, body, todayLessonDay);
              break;
            case 'web':
              success = await sendWebPush(token.device_token, title, body, todayLessonDay);
              break;
            default:
              console.log(`[Push Cron] Unsupported platform: ${token.platform}`);
          }

          if (success) {
            sentCount++;

            // Log the notification
            await supabase.from('notification_log').insert({
              user_id: user.id,
              notification_type: 'daily_reminder',
              title,
              body,
              copy_variant: selectedCopy.variant_code,
              platform: token.platform,
              device_token_id: token.id,
              sent_at: new Date().toISOString(),
              lesson_day: todayLessonDay,
              streak_count: streak,
              metadata: { date_formatted: dateInfo.formatted }
            });
          } else {
            failedCount++;
          }
        } catch (error) {
          failedCount++;
          errors.push(`${user.id}/${token.platform}: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
      }
    }

    console.log(`[Push Cron] Complete: ${sentCount} sent, ${skippedCount} skipped, ${failedCount} failed`);

    return res.status(200).json({
      success: true,
      message: 'Daily push notifications processed',
      stats: {
        totalUsers: users.length,
        sent: sentCount,
        skipped: skippedCount,
        failed: failedCount
      },
      lesson: {
        dayNumber: todayLessonDay, // Internal reference only
        date: dateInfo.formatted,  // User-facing: "December 17"
        title: lesson.title,
        emoji: lesson.emoji
      },
      copyVariant: selectedCopy.variant_code,
      errors: errors.length > 0 ? errors.slice(0, 10) : undefined
    });

  } catch (error) {
    console.error('[Push Cron] Error:', error);
    return res.status(500).json({
      error: 'Failed to process push notifications',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

