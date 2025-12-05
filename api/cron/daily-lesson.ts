/**
 * Daily Lesson Cron Job
 * 
 * Runs every day at 12pm UTC (7am EST / 4am PST)
 * Sends daily lesson emails to all subscribed users.
 * 
 * Triggered by Vercel Cron or external cron service.
 * 
 * Environment Variables:
 * - RESEND_API_KEY: Your Resend API key
 * - CRON_SECRET: Secret to verify cron requests (optional but recommended)
 * - SUPABASE_URL: Your Supabase project URL
 * - SUPABASE_SERVICE_ROLE_KEY: Supabase service role key for fetching users
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

const RESEND_API_URL = 'https://api.resend.com/emails';
const RESEND_BATCH_URL = 'https://api.resend.com/emails/batch';

// Sample lessons - in production, fetch from your database
const DAILY_LESSONS = [
  { day: 1, title: "How Money Works", emoji: "💰", category: "Economics" },
  { day: 2, title: "Why the Sky is Blue", emoji: "🌤️", category: "Science" },
  { day: 3, title: "The Secret Life of Trees", emoji: "🌳", category: "Nature" },
  { day: 4, title: "How Your Phone Knows Where You Are", emoji: "📱", category: "Technology" },
  { day: 5, title: "Why We Dream", emoji: "💭", category: "Psychology" },
  // Add more lessons...
];

const FUN_FACTS = [
  "The average person stops actively learning around age 25. You just changed that!",
  "You create 700 new neural connections every time you learn something new.",
  "Children ask ~300 questions per day. Adults? About 20. Let's change that!",
  "Reading 20 minutes a day exposes you to 1.8 million words per year.",
];

function getRandomFact(): string {
  return FUN_FACTS[Math.floor(Math.random() * FUN_FACTS.length)];
}

function getDayOfYear(): number {
  const now = new Date();
  const start = new Date(now.getFullYear(), 0, 0);
  const diff = now.getTime() - start.getTime();
  const oneDay = 1000 * 60 * 60 * 24;
  return Math.floor(diff / oneDay);
}

function generateDailyLessonHTML(name: string, lesson: typeof DAILY_LESSONS[0], lessonUrl: string): string {
  const fact = getRandomFact();
  return `
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0a0a0b; color: #f4f4f5;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #0a0a0b; padding: 40px 20px;">
    <tr><td align="center">
      <table width="100%" style="max-width: 560px; background-color: #18181b; border-radius: 16px; overflow: hidden;">
        <tr><td style="padding: 40px 40px 20px; text-align: center;">
          <img src="https://curiouskelly.com/images/brand/kelly-mark-circle-64.png" alt="Kelly" width="80" height="80" style="border-radius: 50%; border: 3px solid #3b82f6;">
        </td></tr>
        <tr><td style="padding: 0 40px 30px;">
          <p style="color: #a1a1aa; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">Good morning, ${name}! ☀️</p>
          <p style="color: #a1a1aa; font-size: 16px; line-height: 1.6; margin: 0 0 30px;">Your daily dose of curiosity is ready.</p>
          <table width="100%" style="background-color: #27272a; border-radius: 12px; margin-bottom: 30px;">
            <tr><td style="padding: 24px;">
              <p style="color: #71717a; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 8px;">Day ${lesson.day} · ${lesson.category}</p>
              <h2 style="color: #f4f4f5; font-size: 24px; font-weight: 600; margin: 0 0 16px;">${lesson.emoji} ${lesson.title}</h2>
              <p style="color: #a1a1aa; font-size: 14px; margin: 0;">⏱️ 5 minutes · 🎯 Perfect for your morning</p>
            </td></tr>
          </table>
          <table width="100%" cellpadding="0" cellspacing="0">
            <tr><td align="center" style="padding: 10px 0 30px;">
              <a href="${lessonUrl}" style="display: inline-block; background-color: #3b82f6; color: white; text-decoration: none; padding: 14px 32px; border-radius: 8px; font-weight: 600; font-size: 16px;">Start Today's Lesson →</a>
            </td></tr>
          </table>
          <p style="color: #f4f4f5; font-size: 16px; line-height: 1.6; margin: 0;">✨ Stay curious,<br><strong>Kelly</strong></p>
        </td></tr>
        <tr><td style="padding: 0 40px 40px;">
          <table width="100%" style="background-color: #27272a; border-radius: 12px;">
            <tr><td style="padding: 20px;">
              <p style="color: #fbbf24; font-size: 14px; font-weight: 600; margin: 0 0 8px;">💡 Did you know?</p>
              <p style="color: #a1a1aa; font-size: 14px; line-height: 1.5; margin: 0;">${fact}</p>
            </td></tr>
          </table>
        </td></tr>
        <tr><td style="padding: 30px 40px; border-top: 1px solid #27272a; text-align: center;">
          <p style="color: #52525b; font-size: 12px; margin: 0 0 10px;">✨ Curious Kelly | Learn something new every day</p>
          <p style="color: #52525b; font-size: 12px; margin: 0;"><a href="https://curiouskelly.com" style="color: #52525b;">curiouskelly.com</a> · <a href="https://curiouskelly.com/help" style="color: #52525b;">Help</a></p>
          <p style="color: #3f3f46; font-size: 11px; margin: 15px 0 0;">© 2025 Lesson of the Day PBC</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>
  `.trim();
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Verify cron secret if configured
  const cronSecret = process.env.CRON_SECRET;
  const authHeader = req.headers.authorization;
  
  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const resendApiKey = process.env.RESEND_API_KEY;
  if (!resendApiKey) {
    return res.status(500).json({ error: 'RESEND_API_KEY not configured' });
  }

  try {
    // Get today's lesson
    const dayOfYear = getDayOfYear();
    const lessonIndex = (dayOfYear - 1) % DAILY_LESSONS.length;
    const lesson = DAILY_LESSONS[lessonIndex];
    const lessonUrl = `https://curiouskelly.com/day/${lesson.day}`;

    // In production: Fetch subscribed users from Supabase
    // For now, this is a manual trigger endpoint
    // You would add: const users = await fetchSubscribedUsers();
    
    // Example: Send to a test list (replace with actual user fetch)
    const testUsers = [
      // Add test emails here or fetch from database
      // { email: 'user@example.com', name: 'User' }
    ];

    if (testUsers.length === 0) {
      return res.status(200).json({ 
        message: 'No users to send to. Add users to the list or connect Supabase.',
        lesson: lesson,
        dayOfYear: dayOfYear
      });
    }

    // Batch send (up to 100 at a time)
    const batches = [];
    for (let i = 0; i < testUsers.length; i += 100) {
      batches.push(testUsers.slice(i, i + 100));
    }

    let totalSent = 0;
    for (const batch of batches) {
      const emails = batch.map((user: { email: string; name?: string }) => ({
        from: 'Kelly <hello@curiouskelly.com>',
        to: user.email,
        subject: `${lesson.emoji} Today's lesson: ${lesson.title}`,
        html: generateDailyLessonHTML(user.name || 'friend', lesson, lessonUrl),
        reply_to: 'hello@curiouskelly.com',
      }));

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
      }
    }

    console.log(`Daily lesson cron: Sent ${totalSent} emails for Day ${lesson.day}: ${lesson.title}`);

    return res.status(200).json({ 
      success: true,
      message: `Sent ${totalSent} daily lesson emails`,
      lesson: lesson,
      dayOfYear: dayOfYear
    });

  } catch (error) {
    console.error('Daily lesson cron error:', error);
    return res.status(500).json({ 
      error: 'Failed to send daily emails',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

