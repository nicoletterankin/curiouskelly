/**
 * Enhanced Daily Lesson Cron Job
 * 
 * Runs every day at 12pm UTC (7am EST / 4am PST)
 * 
 * Features:
 * - Personalized progress (streak, total lessons)
 * - Smart subject line rotation
 * - Milestone celebrations (7, 14, 30, 60, 100, 365 days)
 * - Birthday fusion (special email on their birthday)
 * - Gentle return for dormant users (separate cron)
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const RESEND_BATCH_URL = 'https://api.resend.com/emails/batch';
const RESEND_API_URL = 'https://api.resend.com/emails';
const BATCH_SIZE = 100;

// Milestone days that trigger celebration
const MILESTONE_DAYS = [7, 14, 30, 60, 100, 200, 365];

// Subject line styles for rotation
type SubjectStyle = 'emoji' | 'progress' | 'curiosity' | 'time';

function getSubjectStyle(userId: string): SubjectStyle {
  // Deterministic rotation based on user ID + day
  const dayOfYear = getDayOfYear();
  const hash = userId.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
  const styles: SubjectStyle[] = ['emoji', 'progress', 'curiosity', 'time'];
  return styles[(hash + dayOfYear) % styles.length];
}

function generateSubject(
  style: SubjectStyle,
  lessonEmoji: string,
  lessonTitle: string,
  streak: number,
  isBirthday: boolean
): string {
  if (isBirthday) {
    return 'Today is yours ✨';
  }
  
  switch (style) {
    case 'emoji':
      return `${lessonEmoji} ${lessonTitle}`;
    case 'progress':
      return streak > 1 ? `Day ${streak}: ${lessonTitle}` : `${lessonEmoji} ${lessonTitle}`;
    case 'curiosity':
      return 'I found something wonderful today';
    case 'time':
      return `Your 5 minutes of wonder: ${lessonTitle}`;
    default:
      return `${lessonEmoji} ${lessonTitle}`;
  }
}

function getDayOfYear(): number {
  const now = new Date();
  const start = new Date(now.getFullYear(), 0, 0);
  const diff = now.getTime() - start.getTime();
  const oneDay = 1000 * 60 * 60 * 24;
  return Math.floor(diff / oneDay);
}

function isTodayUserBirthday(birthday: string | null): boolean {
  if (!birthday) return false;
  const today = new Date();
  const bday = new Date(birthday);
  return today.getMonth() === bday.getMonth() && today.getDate() === bday.getDate();
}

function getMilestoneMessage(streak: number): string | null {
  if (!MILESTONE_DAYS.includes(streak)) return null;
  
  const messages: Record<number, string> = {
    7: "Seven days in a row. That's not nothing. Most people don't make it past three.",
    14: "Two weeks of learning together. Something's taking root.",
    30: "A whole month. Day after day, you kept coming back. That says something about who you are.",
    60: "60 days. At this point, it's not a streak anymore — it's just what you do.",
    100: "100 days. I don't really know what to say except... thank you. For showing up.",
    200: "200 days. You're extraordinary. Half a year of daily curiosity.",
    365: "A full year. Every single day. You did something most people only dream about."
  };
  
  return messages[streak] || null;
}

/**
 * Generate the enhanced daily lesson email
 */
function generateEnhancedEmailHTML(
  name: string,
  lessonTitle: string,
  lessonEmoji: string,
  dayNumber: number,
  lessonUrl: string,
  unsubscribeUrl: string,
  streak: number,
  totalLessons: number,
  milestoneMessage: string | null,
  isBirthday: boolean,
  birthdayLessonUrl?: string
): string {
  // Build the progress line
  let progressLine = '';
  if (totalLessons > 0 && streak > 1) {
    progressLine = `You've learned ${totalLessons} thing${totalLessons > 1 ? 's' : ''} with me. Here's #${totalLessons + 1}.`;
  }

  // Build milestone section
  let milestoneSection = '';
  if (milestoneMessage && !isBirthday) {
    milestoneSection = `
      <p style="font-family: Georgia, serif; font-size: 17px; color: #4b5563; line-height: 1.9; margin: 0 0 24px; padding: 20px; background: #fef3c7; border-radius: 8px;">
        <strong style="color: #92400e;">Wait — before today's lesson...</strong><br><br>
        ${streak} days. ${milestoneMessage}
      </p>
    `;
  }

  // Build birthday section
  let birthdaySection = '';
  if (isBirthday) {
    birthdaySection = `
      <p style="font-family: Georgia, serif; font-size: 21px; color: #1f2937; line-height: 1.7; margin: 0 0 24px;">
        <strong>Happy birthday.</strong>
      </p>
      <p style="font-family: Georgia, serif; font-size: 17px; color: #4b5563; line-height: 1.9; margin: 0 0 24px;">
        Your birthday lesson is waiting. It's always the same one, every year. But somehow it means something different each time.
      </p>
      ${birthdayLessonUrl ? `
        <p style="margin: 0 0 32px;">
          <a href="${birthdayLessonUrl}" style="display: inline-block; background: #fbbf24; color: #1f2937; padding: 14px 28px; border-radius: 8px; text-decoration: none; font-family: -apple-system, sans-serif; font-size: 15px; font-weight: 500;">
            Your birthday lesson →
          </a>
        </p>
      ` : ''}
      <p style="font-family: Georgia, serif; font-size: 17px; color: #4b5563; line-height: 1.9; margin: 0 0 24px;">
        And when you're ready, today's daily lesson is here too:
      </p>
    `;
  }

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
              <p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; margin: 0 0 20px;">
                Good morning${name !== 'friend' ? ', ' + name : ''}.
              </p>
              
              ${progressLine ? `
                <p style="font-family: Georgia, serif; font-size: 15px; color: #6b7280; line-height: 1.7; margin: 0 0 24px;">
                  ${progressLine}
                </p>
              ` : ''}
              
              ${milestoneSection}
              
              ${birthdaySection}
              
              <p style="font-family: Georgia, serif; font-size: 21px; color: #1f2937; line-height: 1.7; margin: 0 0 8px;">
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

function generateEnhancedEmailText(
  name: string,
  lessonTitle: string,
  lessonEmoji: string,
  dayNumber: number,
  lessonUrl: string,
  unsubscribeUrl: string,
  streak: number,
  totalLessons: number,
  milestoneMessage: string | null,
  isBirthday: boolean
): string {
  let text = `Good morning${name !== 'friend' ? ', ' + name : ''}.\n\n`;
  
  if (totalLessons > 0 && streak > 1) {
    text += `You've learned ${totalLessons} things with me. Here's #${totalLessons + 1}.\n\n`;
  }
  
  if (milestoneMessage && !isBirthday) {
    text += `Wait — before today's lesson...\n\n${streak} days. ${milestoneMessage}\n\n`;
  }
  
  if (isBirthday) {
    text += `Happy birthday.\n\nYour birthday lesson is waiting. It's always the same one, every year. But somehow it means something different each time.\n\nAnd when you're ready, today's daily lesson is here too:\n\n`;
  }
  
  text += `${lessonEmoji} ${lessonTitle}\n\n`;
  text += `Five minutes. I think you'll love it.\n\n`;
  text += `${lessonUrl}\n\n`;
  text += `— Kelly\n\n`;
  text += `---\n`;
  text += `Day ${dayNumber} of 365 · curiouskelly.com\n`;
  text += `Unsubscribe: ${unsubscribeUrl}`;
  
  return text;
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

  if (!resendApiKey) {
    return res.status(500).json({ error: 'RESEND_API_KEY not configured' });
  }

  if (!supabaseUrl || !supabaseServiceKey) {
    return res.status(500).json({ error: 'Supabase not configured' });
  }

  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  try {
    const dayOfYear = getDayOfYear();
    
    // Fetch today's lesson
    let lesson: any = null;
    const { data: lessonData, error: lessonError } = await supabase
      .from('lessons')
      .select('id, title, emoji, category, day_number')
      .eq('day_number', dayOfYear)
      .single();

    if (!lessonError && lessonData) {
      lesson = lessonData;
    } else {
      // Fallback: try any lesson for this day
      const { data: fallback } = await supabase
        .from('lessons')
        .select('id, title, emoji, category, day_number')
        .eq('day_number', dayOfYear)
        .limit(1)
        .single();
      lesson = fallback;
    }

    const lessonEmoji = lesson?.emoji || '📚';
    const lessonTitle = lesson?.title || `Day ${dayOfYear} Lesson`;
    const lessonDayNumber = lesson?.day_number || dayOfYear;
    const lessonUrl = `https://curiouskelly.com/day/${lessonDayNumber}`;

    // Fetch subscribed users with their progress
    const { data: users, error: usersError } = await supabase
      .from('users')
      .select('id, email, display_name, name, unsubscribe_token, birthday, current_streak, total_lessons_completed')
      .eq('email_daily_lesson', true)
      .is('email_unsubscribed_at', null)
      .not('email', 'is', null);

    if (usersError) {
      console.error('Could not fetch users', usersError);
      return res.status(500).json({ error: 'Could not fetch users', details: usersError.message });
    }

    if (!users || users.length === 0) {
      return res.status(200).json({
        success: true,
        message: 'No subscribed users',
        lesson: { day: lessonDayNumber, title: lessonTitle, emoji: lessonEmoji }
      });
    }

    console.log(`Enhanced daily lesson: Sending to ${users.length} users`);

    // Process users and send emails
    let totalSent = 0;
    let totalFailed = 0;
    let birthdayCount = 0;
    let milestoneCount = 0;
    const errors: string[] = [];

    // Send in batches
    for (let i = 0; i < users.length; i += BATCH_SIZE) {
      const batch = users.slice(i, i + BATCH_SIZE);
      
      const emails = batch.map(user => {
        const displayName = user.display_name || user.name || 'friend';
        const unsubscribeUrl = `https://curiouskelly.com/api/unsubscribe?token=${user.unsubscribe_token}`;
        const streak = user.current_streak || 0;
        const totalLessons = user.total_lessons_completed || 0;
        const isBirthday = isTodayUserBirthday(user.birthday);
        const milestoneMessage = getMilestoneMessage(streak + 1); // +1 for today
        
        if (isBirthday) birthdayCount++;
        if (milestoneMessage) milestoneCount++;
        
        // Get birthday lesson URL if it's their birthday
        let birthdayLessonUrl: string | undefined;
        if (isBirthday && user.birthday) {
          const bday = new Date(user.birthday);
          const bdayDayOfYear = Math.floor((bday.getTime() - new Date(bday.getFullYear(), 0, 0).getTime()) / (1000 * 60 * 60 * 24));
          birthdayLessonUrl = `https://curiouskelly.com/day/${bdayDayOfYear}`;
        }
        
        const subjectStyle = getSubjectStyle(user.id);
        const subject = generateSubject(subjectStyle, lessonEmoji, lessonTitle, streak + 1, isBirthday);
        
        return {
          from: 'Kelly <hello@curiouskelly.com>',
          to: user.email,
          subject,
          html: generateEnhancedEmailHTML(
            displayName,
            lessonTitle,
            lessonEmoji,
            lessonDayNumber,
            lessonUrl,
            unsubscribeUrl,
            streak + 1,
            totalLessons,
            milestoneMessage,
            isBirthday,
            birthdayLessonUrl
          ),
          text: generateEnhancedEmailText(
            displayName,
            lessonTitle,
            lessonEmoji,
            lessonDayNumber,
            lessonUrl,
            unsubscribeUrl,
            streak + 1,
            totalLessons,
            milestoneMessage,
            isBirthday
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
        errors.push(`Batch ${Math.floor(i / BATCH_SIZE) + 1}: Network error`);
      }
    }

    console.log(`Enhanced daily lesson complete: ${totalSent} sent, ${totalFailed} failed, ${birthdayCount} birthdays, ${milestoneCount} milestones`);

    return res.status(200).json({
      success: true,
      message: 'Enhanced daily lesson emails sent',
      stats: {
        totalUsers: users.length,
        sent: totalSent,
        failed: totalFailed,
        birthdays: birthdayCount,
        milestones: milestoneCount
      },
      lesson: {
        day: lessonDayNumber,
        title: lessonTitle,
        emoji: lessonEmoji
      },
      errors: errors.length > 0 ? errors : undefined
    });

  } catch (error) {
    console.error('Enhanced daily lesson error:', error);
    return res.status(500).json({
      error: 'Failed to process daily emails',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}
