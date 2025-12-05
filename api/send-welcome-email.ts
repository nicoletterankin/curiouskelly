/**
 * Welcome Email API Endpoint
 * 
 * Sends a personalized welcome email when a user signs up.
 * Triggered by Supabase auth webhook or called directly after signup.
 * 
 * Environment Variables Required:
 * - RESEND_API_KEY: Your Resend API key
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

// Resend API endpoint
const RESEND_API_URL = 'https://api.resend.com/emails';

// Fun facts to rotate in emails
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

function generateWelcomeEmailHTML(name: string, lessonUrl: string): string {
  const funFact = getRandomFact();
  
  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Welcome to Curious Kelly!</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #0a0a0b; color: #f4f4f5;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #0a0a0b; padding: 40px 20px;">
    <tr>
      <td align="center">
        <table width="100%" style="max-width: 560px; background-color: #18181b; border-radius: 16px; overflow: hidden;">
          
          <!-- Header -->
          <tr>
            <td style="padding: 40px 40px 20px; text-align: center;">
              <img src="https://curiouskelly.com/images/brand/kelly-mark-circle-64.png" alt="Kelly" width="80" height="80" style="border-radius: 50%; border: 3px solid #3b82f6;">
            </td>
          </tr>
          
          <!-- Main Content -->
          <tr>
            <td style="padding: 0 40px 30px;">
              <h1 style="color: #f4f4f5; font-size: 28px; font-weight: 600; margin: 0 0 20px; text-align: center;">
                Welcome to curiosity, ${name}! 🎉
              </h1>
              
              <p style="color: #a1a1aa; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">
                Hey ${name}!
              </p>
              
              <p style="color: #a1a1aa; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">
                I'm Kelly, and I'm SO excited you're here.
              </p>
              
              <p style="color: #a1a1aa; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">
                You just joined thousands of curious minds who've decided that every day is a chance to learn something new. That's pretty amazing.
              </p>
              
              <p style="color: #f4f4f5; font-size: 16px; line-height: 1.6; margin: 0 0 10px; font-weight: 600;">
                Here's what happens next:
              </p>
              
              <ul style="color: #a1a1aa; font-size: 16px; line-height: 1.8; margin: 0 0 30px; padding-left: 20px;">
                <li>Every day, I'll have a fresh lesson waiting for you</li>
                <li>Takes about 5 minutes (perfect with your morning coffee ☕)</li>
                <li>You'll finish each one knowing something you didn't before</li>
              </ul>
              
              <!-- CTA Button -->
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td align="center" style="padding: 10px 0 30px;">
                    <a href="${lessonUrl}" style="display: inline-block; background-color: #3b82f6; color: white; text-decoration: none; padding: 14px 32px; border-radius: 8px; font-weight: 600; font-size: 16px;">
                      Start Your First Lesson →
                    </a>
                  </td>
                </tr>
              </table>
              
              <p style="color: #a1a1aa; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">
                Can't wait to learn together!
              </p>
              
              <p style="color: #f4f4f5; font-size: 16px; line-height: 1.6; margin: 0;">
                ✨ Stay curious,<br>
                <strong>Kelly</strong>
              </p>
              
              <p style="color: #71717a; font-size: 14px; line-height: 1.6; margin: 20px 0 0;">
                P.S. Hit reply anytime - I read every message!
              </p>
            </td>
          </tr>
          
          <!-- Fun Fact Box -->
          <tr>
            <td style="padding: 0 40px 40px;">
              <table width="100%" style="background-color: #27272a; border-radius: 12px; padding: 20px;">
                <tr>
                  <td style="padding: 20px;">
                    <p style="color: #fbbf24; font-size: 14px; font-weight: 600; margin: 0 0 8px;">
                      💡 Did you know?
                    </p>
                    <p style="color: #a1a1aa; font-size: 14px; line-height: 1.5; margin: 0;">
                      ${funFact}
                    </p>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          
          <!-- Footer -->
          <tr>
            <td style="padding: 30px 40px; border-top: 1px solid #27272a; text-align: center;">
              <p style="color: #52525b; font-size: 12px; margin: 0 0 10px;">
                Curious Kelly | Learn something new every day
              </p>
              <p style="color: #52525b; font-size: 12px; margin: 0;">
                <a href="https://curiouskelly.com" style="color: #52525b;">curiouskelly.com</a> · 
                <a href="https://curiouskelly.com/help" style="color: #52525b;">Help</a> · 
                <a href="https://curiouskelly.com/privacy" style="color: #52525b;">Privacy</a>
              </p>
              <p style="color: #3f3f46; font-size: 11px; margin: 15px 0 0;">
                © 2025 Lesson of the Day PBC
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

function generateWelcomeEmailText(name: string, lessonUrl: string): string {
  const funFact = getRandomFact();
  
  return `
Welcome to curiosity, ${name}! 🎉

Hey ${name}!

I'm Kelly, and I'm SO excited you're here.

You just joined thousands of curious minds who've decided that every day is a chance to learn something new. That's pretty amazing.

Here's what happens next:
→ Every day, I'll have a fresh lesson waiting for you
→ Takes about 5 minutes (perfect with your morning coffee ☕)
→ You'll finish each one knowing something you didn't before

Start your first lesson: ${lessonUrl}

Can't wait to learn together!

✨ Stay curious,
Kelly

P.S. Hit reply anytime - I read every message!

---
💡 Did you know? ${funFact}

---
Curious Kelly | Learn something new every day
curiouskelly.com

© 2025 Lesson of the Day PBC
  `.trim();
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Verify API key (simple auth)
  const authHeader = req.headers.authorization;
  const expectedKey = process.env.WELCOME_EMAIL_API_KEY;
  
  if (expectedKey && authHeader !== `Bearer ${expectedKey}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  // Get request body
  const { email, name } = req.body;

  if (!email) {
    return res.status(400).json({ error: 'Email is required' });
  }

  // Get Resend API key
  const resendApiKey = process.env.RESEND_API_KEY;
  
  if (!resendApiKey) {
    console.error('RESEND_API_KEY not configured');
    return res.status(500).json({ error: 'Email service not configured' });
  }

  // Default name if not provided
  const displayName = name || 'Curious Friend';
  const lessonUrl = 'https://curiouskelly.com/learn';

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
        subject: `Welcome to curiosity, ${displayName}! 🎉`,
        html: generateWelcomeEmailHTML(displayName, lessonUrl),
        text: generateWelcomeEmailText(displayName, lessonUrl),
        reply_to: 'hello@curiouskelly.com',
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      console.error('Resend API error:', data);
      return res.status(response.status).json({ 
        error: 'Failed to send email',
        details: data 
      });
    }

    console.log(`Welcome email sent to ${email}`, data);
    
    return res.status(200).json({ 
      success: true,
      message: 'Welcome email sent',
      id: data.id 
    });

  } catch (error) {
    console.error('Error sending welcome email:', error);
    return res.status(500).json({ 
      error: 'Failed to send email',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}


