/**
 * Gentle Return Cron Job
 * 
 * Sends warm re-engagement emails to dormant users.
 * Runs daily, checks for users who haven't opened emails in 7+ days.
 * 
 * Kelly's approach: No guilt. No "we miss you" BS. Just warmth.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const RESEND_API_URL = 'https://api.resend.com/emails';

// Days of inactivity before sending gentle return
const DORMANT_THRESHOLD_DAYS = 7;

// Don't send more than one gentle return per 14 days
const GENTLE_RETURN_COOLDOWN_DAYS = 14;

function generateGentleReturnHTML(
  name: string,
  lessonTitle: string,
  lessonEmoji: string,
  lessonUrl: string,
  unsubscribeUrl: string
): string {
  // Kelly's Voice: Humble, no guilt, just warmth
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
                Hey${name !== 'friend' ? ' ' + name : ''}.
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 17px; color: #4b5563; line-height: 1.9; margin: 0 0 24px;">
                I noticed you haven't been by in a while. No guilt, no pressure.
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 17px; color: #4b5563; line-height: 1.9; margin: 0 0 24px;">
                Just wanted you to know: the lessons are always here. Whenever you're ready, I'll be ready too.
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 17px; color: #4b5563; line-height: 1.9; margin: 0 0 32px;">
                Today's lesson is about <strong>${lessonTitle}</strong>. Kind of fitting, right? (Or maybe not. I never know.)
              </p>
              
              <p style="margin: 0 0 32px;">
                <a href="${lessonUrl}" style="display: inline-block; background: #2563eb; color: white; padding: 14px 28px; border-radius: 8px; text-decoration: none; font-family: -apple-system, sans-serif; font-size: 15px; font-weight: 500;">
                  Come back when you're ready →
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
              <p style="font-family: -apple-system, sans-serif; font-size: 11px; color: #9ca3af; margin: 0;">
                <a href="${unsubscribeUrl}" style="color: #9ca3af;">Stop receiving these emails</a>
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

function generateGentleReturnText(
  name: string,
  lessonTitle: string,
  lessonUrl: string,
  unsubscribeUrl: string
): string {
  return `
Hey${name !== 'friend' ? ' ' + name : ''}.

I noticed you haven't been by in a while. No guilt, no pressure.

Just wanted you to know: the lessons are always here. Whenever you're ready, I'll be ready too.

Today's lesson is about ${lessonTitle}. Kind of fitting, right? (Or maybe not. I never know.)

Come back when you're ready: ${lessonUrl}

— Kelly

---
Stop receiving these emails: ${unsubscribeUrl}
  `.trim();
}

function getDayOfYear(): number {
  const now = new Date();
  const start = new Date(now.getFullYear(), 0, 0);
  const diff = now.getTime() - start.getTime();
  const oneDay = 1000 * 60 * 60 * 24;
  return Math.floor(diff / oneDay);
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Verify cron secret
  const cronSecret = process.env.CRON_SECRET;
  const authHeader = req.headers.authorization;
  
  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const resendApiKey = process.env.RESEND_API_KEY;
  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!resendApiKey || !supabaseUrl || !supabaseServiceKey) {
    return res.status(500).json({ error: 'Configuration error' });
  }

  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  try {
    const dayOfYear = getDayOfYear();
    const now = new Date();
    const dormantCutoff = new Date(now.getTime() - DORMANT_THRESHOLD_DAYS * 24 * 60 * 60 * 1000);
    const cooldownCutoff = new Date(now.getTime() - GENTLE_RETURN_COOLDOWN_DAYS * 24 * 60 * 60 * 1000);

    // Get today's lesson for the email
    const { data: lesson } = await supabase
      .from('lessons')
      .select('title, emoji, day_number')
      .eq('day_number', dayOfYear)
      .single();

    const lessonTitle = lesson?.title || 'something wonderful';
    const lessonEmoji = lesson?.emoji || '📚';
    const lessonUrl = `https://curiouskelly.com/day/${lesson?.day_number || dayOfYear}`;

    // Find dormant users:
    // - Subscribed to daily emails
    // - Haven't opened email in 7+ days (or never tracked)
    // - Haven't received a gentle return in 14+ days
    const { data: dormantUsers, error: usersError } = await supabase
      .from('users')
      .select('id, email, display_name, name, unsubscribe_token, last_email_opened_at, last_gentle_return_at')
      .eq('email_daily_lesson', true)
      .is('email_unsubscribed_at', null)
      .not('email', 'is', null)
      .or(`last_email_opened_at.is.null,last_email_opened_at.lt.${dormantCutoff.toISOString()}`)
      .or(`last_gentle_return_at.is.null,last_gentle_return_at.lt.${cooldownCutoff.toISOString()}`);

    if (usersError) {
      console.error('Error fetching dormant users:', usersError);
      return res.status(500).json({ error: 'Database error', details: usersError.message });
    }

    if (!dormantUsers || dormantUsers.length === 0) {
      return res.status(200).json({
        success: true,
        message: 'No dormant users found',
        stats: { checked: 0, sent: 0 }
      });
    }

    console.log(`Gentle return: Found ${dormantUsers.length} dormant users`);

    let sent = 0;
    let failed = 0;
    const errors: string[] = [];

    // Send individual emails (not batch, these are special)
    for (const user of dormantUsers) {
      const displayName = user.display_name || user.name || 'friend';
      const unsubscribeUrl = `https://curiouskelly.com/api/unsubscribe?token=${user.unsubscribe_token}&type=all`;

      try {
        const response = await fetch(RESEND_API_URL, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${resendApiKey}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            from: 'Kelly <hello@curiouskelly.com>',
            to: user.email,
            subject: 'No pressure',
            html: generateGentleReturnHTML(displayName, lessonTitle, lessonEmoji, lessonUrl, unsubscribeUrl),
            text: generateGentleReturnText(displayName, lessonTitle, lessonUrl, unsubscribeUrl),
            reply_to: 'hello@curiouskelly.com',
          }),
        });

        if (response.ok) {
          sent++;
          // Update last_gentle_return_at
          await supabase
            .from('users')
            .update({ last_gentle_return_at: now.toISOString() })
            .eq('id', user.id);
        } else {
          failed++;
          const errorData = await response.json();
          errors.push(`${user.email}: ${errorData.message || 'Unknown error'}`);
        }
      } catch (error) {
        failed++;
        errors.push(`${user.email}: Network error`);
      }
    }

    console.log(`Gentle return complete: ${sent} sent, ${failed} failed`);

    return res.status(200).json({
      success: true,
      message: 'Gentle return emails processed',
      stats: {
        dormantUsers: dormantUsers.length,
        sent,
        failed
      },
      errors: errors.length > 0 ? errors : undefined
    });

  } catch (error) {
    console.error('Gentle return error:', error);
    return res.status(500).json({
      error: 'Failed to process gentle return emails',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

