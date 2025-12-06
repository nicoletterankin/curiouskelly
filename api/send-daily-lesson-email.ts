/**
 * Daily Lesson Email API
 * 
 * Sends daily lesson reminder emails to subscribed users.
 * Can be triggered by:
 * - Cron job (recommended: daily at 7am in user's timezone)
 * - Manual trigger for testing
 * - Batch processing for all users
 * 
 * Environment Variables:
 * - RESEND_API_KEY: Your Resend API key
 * - DAILY_EMAIL_API_KEY: Secret key to authorize daily email sends
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

const RESEND_API_URL = 'https://api.resend.com/emails';
const RESEND_BATCH_URL = 'https://api.resend.com/emails/batch';

// Brand colors
const COLORS = {
  background: '#0a0a0b',
  cardBg: '#18181b',
  accent: '#3b82f6',
  gold: '#fbbf24',
  text: '#f4f4f5',
  textMuted: '#a1a1aa',
  textDim: '#71717a',
  border: '#27272a',
};

const FUN_FACTS = [
  "The average person stops actively learning around age 25. You just changed that!",
  "You create 700 new neural connections every time you learn something new.",
  "Children ask ~300 questions per day. Adults? About 20. Let's change that!",
  "Reading 20 minutes a day exposes you to 1.8 million words per year.",
  "Einstein failed his university entrance exam the first time. Look where curiosity took him!",
  "Your brain uses 20% of your body's energy but is only 2% of your weight.",
  "The word 'curious' comes from Latin 'cura' meaning 'care' - to be curious is to care!",
  "It takes an average of 66 days to form a habit. Day 1 starts now!",
];

function getRandomFact(): string {
  return FUN_FACTS[Math.floor(Math.random() * FUN_FACTS.length)];
}

function generateDailyLessonHTML(
  name: string,
  lessonTitle: string,
  lessonEmoji: string,
  lessonCategory: string,
  dayNumber: number,
  lessonUrl: string
): string {
  const fact = getRandomFact();
  
  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: ${COLORS.background}; color: ${COLORS.text};">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: ${COLORS.background}; padding: 40px 20px;">
    <tr>
      <td align="center">
        <table width="100%" style="max-width: 560px; background-color: ${COLORS.cardBg}; border-radius: 16px; overflow: hidden;">
          
          <!-- Header -->
          <tr>
            <td style="padding: 40px 40px 20px; text-align: center;">
              <img src="https://curiouskelly.com/images/brand/kelly-mark-circle-64.png" alt="Kelly" width="80" height="80" style="border-radius: 50%; border: 3px solid ${COLORS.accent};">
            </td>
          </tr>
          
          <!-- Main Content -->
          <tr>
            <td style="padding: 0 40px 30px;">
              <p style="color: ${COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">
                Good morning, ${name}! ☀️
              </p>
              <p style="color: ${COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 30px;">
                Your daily dose of curiosity is ready.
              </p>
              
              <!-- Lesson Card -->
              <table width="100%" style="background-color: ${COLORS.border}; border-radius: 12px; margin-bottom: 30px;">
                <tr>
                  <td style="padding: 24px;">
                    <p style="color: ${COLORS.textDim}; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 8px;">
                      Day ${dayNumber} · ${lessonCategory}
                    </p>
                    <h2 style="color: ${COLORS.text}; font-size: 24px; font-weight: 600; margin: 0 0 16px;">
                      ${lessonEmoji} ${lessonTitle}
                    </h2>
                    <p style="color: ${COLORS.textMuted}; font-size: 14px; margin: 0;">
                      ⏱️ 5 minutes · 🎯 Perfect for your morning
                    </p>
                  </td>
                </tr>
              </table>
              
              <!-- CTA Button -->
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td align="center" style="padding: 10px 0 30px;">
                    <a href="${lessonUrl}" style="display: inline-block; background-color: ${COLORS.accent}; color: white; text-decoration: none; padding: 14px 32px; border-radius: 8px; font-weight: 600; font-size: 16px;">
                      Start Today's Lesson →
                    </a>
                  </td>
                </tr>
              </table>
              
              <p style="color: ${COLORS.text}; font-size: 16px; line-height: 1.6; margin: 0;">
                ✨ Stay curious,<br><strong>Kelly</strong>
              </p>
            </td>
          </tr>
          
          <!-- Fun Fact Box -->
          <tr>
            <td style="padding: 0 40px 40px;">
              <table width="100%" style="background-color: ${COLORS.border}; border-radius: 12px;">
                <tr>
                  <td style="padding: 20px;">
                    <p style="color: ${COLORS.gold}; font-size: 14px; font-weight: 600; margin: 0 0 8px;">
                      💡 Did you know?
                    </p>
                    <p style="color: ${COLORS.textMuted}; font-size: 14px; line-height: 1.5; margin: 0;">
                      ${fact}
                    </p>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          
          <!-- Footer -->
          <tr>
            <td style="padding: 30px 40px; border-top: 1px solid ${COLORS.border}; text-align: center;">
              <p style="color: #52525b; font-size: 12px; margin: 0 0 10px;">
                Day ${dayNumber} of 365 · <a href="https://curiouskelly.com" style="color: #52525b;">curiouskelly.com</a>
              </p>
              <p style="color: #3f3f46; font-size: 11px; margin: 10px 0 0;">
                <a href="https://curiouskelly.com/api/unsubscribe?token=UNSUBSCRIBE_TOKEN" style="color: #3f3f46;">Unsubscribe</a>
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
  lessonCategory: string,
  dayNumber: number,
  lessonUrl: string
): string {
  const fact = getRandomFact();
  
  return `
Good morning, ${name}! ☀️

Your daily dose of curiosity is ready.

═══════════════════════════════
Day ${dayNumber} · ${lessonCategory}

${lessonEmoji} ${lessonTitle}

⏱️ 5 minutes · 🎯 Perfect for your morning
═══════════════════════════════

Start today's lesson: ${lessonUrl}

✨ Stay curious,
Kelly

---
💡 Did you know? ${fact}

---
✨ Curious Kelly | Learn something new every day
curiouskelly.com | © 2025 Lesson of the Day PBC
  `.trim();
}

interface DailyLessonRequest {
  email?: string;
  name?: string;
  lessonTitle: string;
  lessonEmoji: string;
  lessonCategory: string;
  dayNumber: number;
  lessonUrl?: string;
  batch?: Array<{
    email: string;
    name?: string;
  }>;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const authHeader = req.headers.authorization;
  const expectedKey = process.env.DAILY_EMAIL_API_KEY;
  
  if (expectedKey && authHeader !== `Bearer ${expectedKey}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const resendApiKey = process.env.RESEND_API_KEY;
  if (!resendApiKey) {
    return res.status(500).json({ error: 'Email service not configured' });
  }

  try {
    const body = req.body as DailyLessonRequest;
    const { lessonTitle, lessonEmoji, lessonCategory, dayNumber, lessonUrl } = body;
    
    if (!lessonTitle || !lessonEmoji || !lessonCategory || !dayNumber) {
      return res.status(400).json({ 
        error: 'Missing required lesson fields',
        required: ['lessonTitle', 'lessonEmoji', 'lessonCategory', 'dayNumber']
      });
    }
    
    const finalLessonUrl = lessonUrl || `https://curiouskelly.com/day/${dayNumber}`;

    // Handle batch emails
    if (body.batch && Array.isArray(body.batch) && body.batch.length > 0) {
      if (body.batch.length > 100) {
        return res.status(400).json({ error: 'Batch size exceeds maximum of 100' });
      }

      const emails = body.batch.map(user => ({
        from: 'Kelly <hello@curiouskelly.com>',
        to: user.email,
        subject: `${lessonEmoji} Today's lesson: ${lessonTitle}`,
        html: generateDailyLessonHTML(user.name || 'friend', lessonTitle, lessonEmoji, lessonCategory, dayNumber, finalLessonUrl),
        text: generateDailyLessonText(user.name || 'friend', lessonTitle, lessonEmoji, lessonCategory, dayNumber, finalLessonUrl),
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

      const data = await response.json();
      if (!response.ok) {
        return res.status(500).json({ error: 'Failed to send batch emails', details: data });
      }

      return res.status(200).json({ 
        success: true,
        message: `Sent ${body.batch.length} daily lesson emails`,
        data: data.data,
      });
    }

    // Handle single email
    if (!body.email) {
      return res.status(400).json({ error: 'Email is required (or provide batch array)' });
    }

    const html = generateDailyLessonHTML(body.name || 'friend', lessonTitle, lessonEmoji, lessonCategory, dayNumber, finalLessonUrl);
    const text = generateDailyLessonText(body.name || 'friend', lessonTitle, lessonEmoji, lessonCategory, dayNumber, finalLessonUrl);

    const response = await fetch(RESEND_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${resendApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'Kelly <hello@curiouskelly.com>',
        to: body.email,
        subject: `${lessonEmoji} Today's lesson: ${lessonTitle}`,
        html,
        text,
        reply_to: 'hello@curiouskelly.com',
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      return res.status(500).json({ error: 'Failed to send email', details: data });
    }

    return res.status(200).json({ 
      success: true,
      message: 'Daily lesson email sent',
      id: data.id,
    });

  } catch (error) {
    console.error('Error sending daily lesson email:', error);
    return res.status(500).json({ 
      error: 'Failed to send email',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}
