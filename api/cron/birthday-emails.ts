import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const resendApiKey = process.env.RESEND_API_KEY;

/**
 * Get the day of year from a date
 */
function getDayOfYear(date: Date): number {
  const start = new Date(date.getFullYear(), 0, 0);
  const diff = date.getTime() - start.getTime();
  const oneDay = 1000 * 60 * 60 * 24;
  return Math.floor(diff / oneDay);
}

/**
 * Generate birthday email HTML with Kelly's voice
 */
function generateBirthdayEmailHTML(name: string, viewCount: number, birthdayLessonDay: number): string {
  const lessonUrl = `https://curiouskelly.com/day/${birthdayLessonDay}`;
  
  const message = viewCount > 1
    ? `Your birthday lesson is waiting. You've learned it ${viewCount} times now. Each year, it means something a little different.`
    : `Your birthday lesson is waiting. This lesson is yours. It always will be.`;

  // Kelly's Voice: Humble, Curious, Collaborative, Warm, Simple, Rich
  return `
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

${name} —<br><br>

Today is yours.<br><br>

${message}<br><br>

<a href="${lessonUrl}" style="color: #1e3a5f; text-decoration: underline;">Learn it again today.</a><br><br>

I hope your year is filled with wonder.<br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
  `.trim();
}

function generateBirthdayEmailText(name: string, viewCount: number, birthdayLessonDay: number): string {
  const lessonUrl = `https://curiouskelly.com/day/${birthdayLessonDay}`;
  
  const message = viewCount > 1
    ? `Your birthday lesson is waiting. You've learned it ${viewCount} times now. Each year, it means something a little different.`
    : `Your birthday lesson is waiting. This lesson is yours. It always will be.`;

  return `
${name} —

Today is yours.

${message}

Learn it again today: ${lessonUrl}

I hope your year is filled with wonder.

— Kelly
  `.trim();
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Verify cron secret
  const cronSecret = process.env.CRON_SECRET;
  const authHeader = req.headers.authorization;
  
  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  if (!supabaseUrl || !supabaseServiceKey) {
    return res.status(500).json({ error: 'Database configuration error' });
  }

  if (!resendApiKey) {
    return res.status(500).json({ error: 'Email configuration error' });
  }

  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  try {
    const today = new Date();
    const todayMonth = today.getMonth() + 1; // 1-12
    const todayDay = today.getDate();
    
    // Find users with birthday today
    // Birthday is stored as DATE, so we extract month and day
    const { data: birthdayUsers, error: fetchError } = await supabase
      .from('users')
      .select('id, email, name, display_name, birthday')
      .not('birthday', 'is', null)
      .not('email', 'is', null);

    if (fetchError) {
      console.error('Error fetching users:', fetchError);
      return res.status(500).json({ error: 'Failed to fetch users' });
    }

    // Filter users whose birthday is today
    const todaysBirthdays = (birthdayUsers || []).filter(user => {
      if (!user.birthday) return false;
      const bday = new Date(user.birthday);
      return bday.getMonth() + 1 === todayMonth && bday.getDate() === todayDay;
    });

    console.log(`Found ${todaysBirthdays.length} users with birthday today`);

    const results = {
      total: todaysBirthdays.length,
      sent: 0,
      failed: 0,
      errors: [] as string[]
    };

    for (const user of todaysBirthdays) {
      try {
        // Get their birthday lesson day
        const birthdayLessonDay = getDayOfYear(new Date(user.birthday!));
        
        // Get view count for this lesson
        const { count: viewCount } = await supabase
          .from('lesson_history')
          .select('*', { count: 'exact', head: true })
          .eq('user_id', user.id)
          .eq('lesson_day', birthdayLessonDay);

        const displayName = user.display_name || user.name || 'friend';

        // Send birthday email via Resend
        const emailResponse = await fetch('https://api.resend.com/emails', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${resendApiKey}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            from: 'Kelly <hello@curiouskelly.com>',
            to: user.email,
            subject: `Happy birthday, ${displayName}`,
            html: generateBirthdayEmailHTML(displayName, viewCount || 0, birthdayLessonDay),
            text: generateBirthdayEmailText(displayName, viewCount || 0, birthdayLessonDay),
            reply_to: 'hello@curiouskelly.com',
          }),
        });

        if (emailResponse.ok) {
          results.sent++;
          console.log(`Birthday email sent to ${user.email}`);
        } else {
          const errorData = await emailResponse.json();
          results.failed++;
          results.errors.push(`${user.email}: ${errorData.message || 'Unknown error'}`);
        }
      } catch (error) {
        results.failed++;
        results.errors.push(`${user.email}: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }

    return res.status(200).json({
      success: true,
      message: `Birthday emails processed`,
      results
    });

  } catch (error) {
    console.error('Error in birthday-emails cron:', error);
    return res.status(500).json({ 
      error: 'Failed to process birthday emails',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}


