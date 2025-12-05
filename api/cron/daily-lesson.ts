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
  // Kelly's Voice: Humble, Curious, Collaborative, Warm, Simple, Rich
  return `
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

Good morning.<br><br>

I found something wonderful today: <strong>${lesson.title}</strong><br><br>

Five minutes. I think you'll love it.<br><br>

<a href="${lessonUrl}" style="color: #1e3a5f; text-decoration: underline;">Let's learn together.</a><br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
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

