/**
 * Email Subscription API
 * 
 * Subscribes a user to daily lesson emails.
 * Creates/updates user record and sends welcome email.
 * 
 * POST /api/subscribe-email
 * Body: { name: string, email: string }
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

const RESEND_API_URL = 'https://api.resend.com/emails';

function generateWelcomeHTML(name: string): string {
  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; background: #fafafa; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background: #fafafa;">
    <tr>
      <td align="center" style="padding: 40px 20px;">
        <table width="100%" style="max-width: 480px;">
          <tr>
            <td style="text-align: center; padding-bottom: 24px;">
              <img src="https://curiouskelly.com/images/brand/kelly-mark-circle-128.png" alt="Kelly" width="80" height="80" style="border-radius: 50%; border: 3px solid #3b82f6;">
            </td>
          </tr>
          <tr>
            <td>
              <p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; margin: 0 0 20px;">
                Hi ${name} — I'm Kelly.
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 17px; color: #374151; line-height: 1.9; margin: 0 0 20px;">
                I don't have all the answers. But I love finding them. And I think learning is better together.
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 17px; color: #374151; line-height: 1.9; margin: 0 0 20px;">
                Starting tomorrow, you'll get a daily email with:
              </p>
              
              <ul style="font-family: Georgia, serif; font-size: 16px; color: #374151; line-height: 1.9; margin: 0 0 20px; padding-left: 20px;">
                <li>🔮 <strong>Learn</strong> — A fascinating topic in 5 minutes</li>
                <li>🎯 <strong>Grow</strong> — A practical challenge to build skills</li>
                <li>🎬 <strong>Watch</strong> — A 100-second video summary</li>
              </ul>
              
              <p style="font-family: Georgia, serif; font-size: 17px; color: #374151; line-height: 1.9; margin: 0 0 24px;">
                But why wait? Today's lesson is ready now.
              </p>
              
              <p style="text-align: center; margin: 0 0 24px;">
                <a href="https://curiouskelly.com/day/351" style="display: inline-block; background: #3b82f6; color: white; padding: 14px 28px; border-radius: 8px; text-decoration: none; font-family: -apple-system, sans-serif; font-size: 16px; font-weight: 600;">
                  Start today's lesson →
                </a>
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 15px; color: #6b7280; font-style: italic; margin: 0;">
                — Kelly
              </p>
            </td>
          </tr>
          <tr>
            <td style="padding-top: 32px; border-top: 1px solid #e5e7eb; margin-top: 32px;">
              <p style="font-family: -apple-system, sans-serif; font-size: 12px; color: #9ca3af; text-align: center; margin: 0;">
                ✨ Curious Kelly · <a href="https://curiouskelly.com" style="color: #9ca3af;">curiouskelly.com</a><br>
                Lesson of the Day PBC · hello@curiouskelly.com
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

function generateWelcomeText(name: string): string {
  return `
Hi ${name} — I'm Kelly.

I don't have all the answers. But I love finding them. And I think learning is better together.

Starting tomorrow, you'll get a daily email with:
- 🔮 Learn — A fascinating topic in 5 minutes
- 🎯 Grow — A practical challenge to build skills
- 🎬 Watch — A 100-second video summary

But why wait? Today's lesson is ready now:
https://curiouskelly.com/day/351

— Kelly

---
✨ Curious Kelly · curiouskelly.com
Lesson of the Day PBC · hello@curiouskelly.com
  `.trim();
}

interface SubscribeRequest {
  name: string;
  email: string;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Handle CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const resendApiKey = process.env.RESEND_API_KEY;
  if (!resendApiKey) {
    console.error('RESEND_API_KEY not configured');
    return res.status(500).json({ error: 'Email service not configured' });
  }

  try {
    const body = req.body as SubscribeRequest;
    const { name, email } = body;
    
    if (!name || !email) {
      return res.status(400).json({ error: 'Name and email are required' });
    }
    
    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return res.status(400).json({ error: 'Invalid email address' });
    }
    
    // TODO: Add to Supabase users table with email_daily_lesson = true
    // For now, just send the welcome email
    
    console.log(`[subscribe-email] New subscriber: ${name} <${email}>`);
    
    // Send welcome email
    const response = await fetch(RESEND_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${resendApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'Kelly <hello@curiouskelly.com>',
        to: email,
        subject: "Welcome! Your daily lessons start tomorrow ✨",
        html: generateWelcomeHTML(name),
        text: generateWelcomeText(name),
        reply_to: 'hello@curiouskelly.com',
      }),
    });

    const data = await response.json();
    
    if (!response.ok) {
      console.error('Resend error:', data);
      return res.status(500).json({ error: 'Failed to send welcome email' });
    }

    console.log(`[subscribe-email] Welcome email sent: ${data.id}`);
    
    return res.status(200).json({ 
      success: true,
      message: 'Subscribed successfully',
      id: data.id,
    });

  } catch (error) {
    console.error('Error subscribing:', error);
    return res.status(500).json({ 
      error: 'Failed to subscribe',
    });
  }
}
