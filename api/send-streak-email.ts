/**
 * Streak Celebration Email API Endpoint
 * 
 * Sends celebration emails for learning streaks (7 days, 30 days, etc.)
 * 
 * Environment Variables Required:
 * - RESEND_API_KEY: Your Resend API key
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

const RESEND_API_URL = 'https://api.resend.com/emails';

// Kelly's Voice: Humble, Curious, Collaborative, Warm, Simple, Rich
const STREAK_MESSAGES: Record<number, { emoji: string; message: string; fact: string }> = {
  7: {
    emoji: '✨',
    message: "Seven days in a row. That's not nothing. Most people don't make it past three. You showed up, and that matters.",
    fact: "Your brain forms new connections every time you learn something. Seven days means seven new pathways."
  },
  14: {
    emoji: '🌱',
    message: "Two weeks of learning together. Something's taking root. I can feel it.",
    fact: "It takes about 66 days to form a habit. You're well on your way."
  },
  30: {
    emoji: '🌿',
    message: "A whole month. Day after day, you kept coming back. That says something about who you are.",
    fact: "30 days of 5-minute lessons adds up to 2.5 hours. That's more focused learning than most adults do in a year."
  },
  60: {
    emoji: '🌳',
    message: "60 days. At this point, it's not a streak anymore — it's just what you do. I'm honored to be part of your routine.",
    fact: "Researchers say habits become automatic around day 66. You're nearly there."
  },
  100: {
    emoji: '💫',
    message: "100 days. I don't really know what to say except... thank you. For trusting me. For showing up. For being curious.",
    fact: "100 different topics. 100 chances to see the world differently. You took every one."
  },
  365: {
    emoji: '🌟',
    message: "A full year. Every single day. I've been thinking about what to write here for a while now. Words don't quite capture it. You did something extraordinary — not because it was hard, but because you chose it. Every day. Thank you for learning with me.",
    fact: "365 lessons. A whole year of wonder. This is what lifelong learning looks like."
  }
};

function generateStreakEmailHTML(name: string, streak: number): string {
  const data = STREAK_MESSAGES[streak] || {
    emoji: '🎉',
    message: `${streak} days of learning! You're building an incredible habit.`,
    fact: `Every day you learn is a day you grow!`
  };
  
  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0a0a0b; color: #f4f4f5;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #0a0a0b; padding: 40px 20px;">
    <tr>
      <td align="center">
        <table width="100%" style="max-width: 560px; background-color: #18181b; border-radius: 16px; overflow: hidden;">
          
          <!-- Celebration Header -->
          <tr>
            <td style="padding: 50px 40px 30px; text-align: center; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);">
              <div style="font-size: 64px; margin-bottom: 16px;">${data.emoji}</div>
              <h1 style="color: white; font-size: 32px; font-weight: 700; margin: 0;">
                ${streak} Day${streak > 1 ? 's' : ''} of Curiosity!
              </h1>
            </td>
          </tr>
          
          <!-- Main Content -->
          <tr>
            <td style="padding: 40px;">
              <p style="font-family: Georgia, serif; color: #f4f4f5; font-size: 18px; margin: 0 0 20px;">
                ${name} —
              </p>
              
              <p style="font-family: Georgia, serif; color: #a1a1aa; font-size: 17px; line-height: 1.8; margin: 0 0 30px;">
                ${data.message}
              </p>
              
              <!-- CTA Button -->
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td align="center" style="padding: 10px 0 30px;">
                    <a href="https://curiouskelly.com/learn" style="display: inline-block; background-color: #2563eb; color: white; text-decoration: none; padding: 14px 28px; border-radius: 8px; font-weight: 500; font-size: 15px; font-family: -apple-system, sans-serif;">
                      Today's lesson →
                    </a>
                  </td>
                </tr>
              </table>
              
              <p style="font-family: Georgia, serif; color: #71717a; font-size: 15px; font-style: italic; margin: 0;">
                — Kelly
              </p>
            </td>
          </tr>
          
          <!-- Fun Fact -->
          <tr>
            <td style="padding: 0 40px 40px;">
              <table width="100%" style="background-color: #27272a; border-radius: 12px;">
                <tr>
                  <td style="padding: 20px;">
                    <p style="color: #fbbf24; font-size: 14px; font-weight: 600; margin: 0 0 8px;">
                      💡 Fun fact:
                    </p>
                    <p style="color: #a1a1aa; font-size: 14px; line-height: 1.5; margin: 0;">
                      ${data.fact}
                    </p>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          
          <!-- Footer -->
          <tr>
            <td style="padding: 30px 40px; border-top: 1px solid #27272a; text-align: center;">
              <p style="color: #52525b; font-size: 12px; margin: 0;">
                <a href="https://curiouskelly.com" style="color: #52525b;">curiouskelly.com</a> · 
                <a href="https://curiouskelly.com/help" style="color: #52525b;">Help</a>
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

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const authHeader = req.headers.authorization;
  const expectedKey = process.env.WELCOME_EMAIL_API_KEY;
  
  if (expectedKey && authHeader !== `Bearer ${expectedKey}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const { email, name, streak } = req.body;

  if (!email || !streak) {
    return res.status(400).json({ error: 'Email and streak are required' });
  }

  const resendApiKey = process.env.RESEND_API_KEY;
  
  if (!resendApiKey) {
    return res.status(500).json({ error: 'Email service not configured' });
  }

  const displayName = name || 'Curious Friend';
  const data = STREAK_MESSAGES[streak] || { emoji: '🎉' };

  try {
    const response = await fetch(RESEND_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${resendApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'Kelly <hello@curiouskelly.com>',
        to: email,
        subject: `${streak} days of curiosity! ${data.emoji}`,
        html: generateStreakEmailHTML(displayName, streak),
        reply_to: 'hello@curiouskelly.com',
      }),
    });

    const result = await response.json();

    if (!response.ok) {
      return res.status(response.status).json({ error: 'Failed to send email', details: result });
    }

    return res.status(200).json({ success: true, id: result.id });

  } catch (error) {
    console.error('Error sending streak email:', error);
    return res.status(500).json({ error: 'Failed to send email' });
  }
}


