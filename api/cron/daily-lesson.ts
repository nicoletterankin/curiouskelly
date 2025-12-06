/**
 * Daily Lesson Cron Job
 * 
 * Runs every day at 12pm UTC (7am EST / 4am PST)
 * Sends daily lesson emails to all subscribed users.
 * 
 * Triggered by Vercel Cron.
 * 
 * Environment Variables:
 * - RESEND_API_KEY: Resend API key
 * - CRON_SECRET: Secret to verify cron requests
 * - PUBLIC_SUPABASE_URL / NEXT_PUBLIC_SUPABASE_URL: Supabase URL
 * - SUPABASE_SERVICE_ROLE_KEY: Supabase service role key
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const RESEND_BATCH_URL = 'https://api.resend.com/emails/batch';
const BATCH_SIZE = 100; // Resend batch limit

/**
 * Get the day of year (1-365)
 */
function getDayOfYear(): number {
  const now = new Date();
  const start = new Date(now.getFullYear(), 0, 0);
  const diff = now.getTime() - start.getTime();
  const oneDay = 1000 * 60 * 60 * 24;
  return Math.floor(diff / oneDay);
}

/**
 * Generate daily lesson email HTML - Kelly's Voice
 */
function generateDailyLessonHTML(
  name: string,
  lessonTitle: string,
  lessonEmoji: string,
  dayNumber: number,
  lessonUrl: string,
  unsubscribeUrl: string
): string {
  // Kelly's Voice: Humble, Curious, Collaborative, Warm, Simple, Rich
  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; background-color: #fafafa;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #fafafa; padding: 40px 20px;">
    <tr>
      <td align="center">
        <table width="100%" style="max-width: 480px;">
          <tr>
            <td style="padding: 32px 24px;">
              <p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; margin: 0 0 24px;">
                Good morning${name !== 'friend' ? ', ' + name : ''}.
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 17px; color: #4b5563; line-height: 1.9; margin: 0 0 24px;">
                I found something wonderful today:
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 21px; color: #1f2937; line-height: 1.7; margin: 0 0 24px;">
                <strong>${lessonEmoji} ${lessonTitle}</strong>
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 17px; color: #4b5563; line-height: 1.9; margin: 0 0 32px;">
                Five minutes. I think you'll love it.
              </p>
              
              <p style="margin: 0 0 32px;">
                <a href="${lessonUrl}" style="display: inline-block; background: #2563eb; color: white; padding: 14px 28px; border-radius: 8px; text-decoration: none; font-family: -apple-system, sans-serif; font-size: 15px; font-weight: 500;">
                  Let's learn together →
                </a>
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 15px; color: #6b7280; font-style: italic; margin: 0;">
                — Kelly
              </p>
            </td>
          </tr>
          
          <!-- Footer -->
          <tr>
            <td style="padding: 24px; text-align: center; border-top: 1px solid #e5e7eb;">
              <p style="font-family: -apple-system, sans-serif; font-size: 12px; color: #9ca3af; margin: 0 0 8px;">
                Day ${dayNumber} of 365 · <a href="https://curiouskelly.com" style="color: #9ca3af;">curiouskelly.com</a>
              </p>
              <p style="font-family: -apple-system, sans-serif; font-size: 11px; color: #9ca3af; margin: 0;">
                <a href="${unsubscribeUrl}" style="color: #9ca3af;">Unsubscribe from daily emails</a>
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
  `.trim();
}

function generateDailyLessonText(
  name: string,
  lessonTitle: string,
  lessonEmoji: string,
  dayNumber: number,
  lessonUrl: string,
  unsubscribeUrl: string
): string {
  return `
Good morning${name !== 'friend' ? ', ' + name : ''}.

I found something wonderful today:

${lessonEmoji} ${lessonTitle}

Five minutes. I think you'll love it.

${lessonUrl}

— Kelly

---
Day ${dayNumber} of 365 · curiouskelly.com
Unsubscribe: ${unsubscribeUrl}
  `.trim();
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Verify cron secret
  const cronSecret = process.env.CRON_SECRET;
  const authHeader = req.headers.authorization;
  
  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  // Check environment variables
  const resendApiKey = process.env.RESEND_API_KEY;
  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!resendApiKey) {
    return res.status(500).json({ error: 'RESEND_API_KEY not configured' });
  }

  if (!supabaseUrl || !supabaseServiceKey) {
    return res.status(500).json({ error: 'Supabase not configured' });
  }

  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  try {
    const dayOfYear = getDayOfYear();
    
    // Fetch today's lesson from database
    const { data: lesson, error: lessonError } = await supabase
      .from('lessons')
      .select('id, title, emoji, category, day_number')
      .eq('day_number', dayOfYear)
      .single();

    if (lessonError || !lesson) {
      console.error('Could not fetch lesson for day', dayOfYear, lessonError);
      return res.status(500).json({ 
        error: 'Could not fetch today\'s lesson',
        dayOfYear,
        details: lessonError?.message
      });
    }

    const lessonUrl = `https://curiouskelly.com/day/${lesson.day_number}`;

    // Fetch all users who want daily emails
    const { data: users, error: usersError } = await supabase
      .from('users')
      .select('id, email, display_name, name, unsubscribe_token')
      .eq('email_daily_lesson', true)
      .is('email_unsubscribed_at', null)
      .not('email', 'is', null);

    if (usersError) {
      console.error('Could not fetch users', usersError);
      return res.status(500).json({ 
        error: 'Could not fetch subscribed users',
        details: usersError.message
      });
    }

    if (!users || users.length === 0) {
      return res.status(200).json({
        success: true,
        message: 'No subscribed users to send to',
        lesson: {
          day: lesson.day_number,
          title: lesson.title,
          emoji: lesson.emoji
        }
      });
    }

    console.log(`Daily lesson cron: Sending to ${users.length} users`);

    // Send in batches
    let totalSent = 0;
    let totalFailed = 0;
    const errors: string[] = [];

    for (let i = 0; i < users.length; i += BATCH_SIZE) {
      const batch = users.slice(i, i + BATCH_SIZE);
      
      const emails = batch.map(user => {
        const displayName = user.display_name || user.name || 'friend';
        const unsubscribeUrl = `https://curiouskelly.com/api/unsubscribe?token=${user.unsubscribe_token}`;
        
        return {
          from: 'Kelly <hello@curiouskelly.com>',
          to: user.email,
          subject: `${lesson.emoji} ${lesson.title}`,
          html: generateDailyLessonHTML(
            displayName,
            lesson.title,
            lesson.emoji,
            lesson.day_number,
            lessonUrl,
            unsubscribeUrl
          ),
          text: generateDailyLessonText(
            displayName,
            lesson.title,
            lesson.emoji,
            lesson.day_number,
            lessonUrl,
            unsubscribeUrl
          ),
          reply_to: 'hello@curiouskelly.com',
        };
      });

      try {
        const response = await fetch(RESEND_BATCH_URL, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${resendApiKey}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(emails),
        });

        if (response.ok) {
          totalSent += batch.length;
        } else {
          const errorData = await response.json();
          totalFailed += batch.length;
          errors.push(`Batch ${Math.floor(i / BATCH_SIZE) + 1}: ${errorData.message || 'Unknown error'}`);
        }
      } catch (batchError) {
        totalFailed += batch.length;
        errors.push(`Batch ${Math.floor(i / BATCH_SIZE) + 1}: ${batchError instanceof Error ? batchError.message : 'Network error'}`);
      }
    }

    console.log(`Daily lesson cron complete: ${totalSent} sent, ${totalFailed} failed`);

    return res.status(200).json({
      success: true,
      message: `Daily lesson emails sent`,
      stats: {
        totalUsers: users.length,
        sent: totalSent,
        failed: totalFailed,
      },
      lesson: {
        day: lesson.day_number,
        title: lesson.title,
        emoji: lesson.emoji,
        category: lesson.category
      },
      errors: errors.length > 0 ? errors : undefined
    });

  } catch (error) {
    console.error('Daily lesson cron error:', error);
    return res.status(500).json({
      error: 'Failed to process daily emails',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}
