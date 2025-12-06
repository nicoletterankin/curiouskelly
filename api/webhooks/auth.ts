/**
 * Supabase Auth Webhook Handler
 * 
 * Receives auth events from Supabase and triggers appropriate actions.
 * Configure in Supabase Dashboard → Auth → Webhooks
 * 
 * Events:
 * - user.created → Send welcome email
 * - user.updated → Handle profile changes
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const RESEND_API_URL = 'https://api.resend.com/emails';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const resendApiKey = process.env.RESEND_API_KEY;
const webhookSecret = process.env.SUPABASE_WEBHOOK_SECRET;

function generateWelcomeEmailHTML(name: string, dayNumber: number, todayLesson: { title: string; emoji: string }): string {
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
              <p style="font-family: Georgia, serif; font-size: 21px; color: #1f2937; line-height: 1.7; margin: 0 0 24px;">
                Hi${name ? ' ' + name : ''},
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 17px; color: #4b5563; line-height: 1.9; margin: 0 0 24px;">
                I'm Kelly. I'm going to teach you one thing, every single day, for the rest of your life.
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 17px; color: #4b5563; line-height: 1.9; margin: 0 0 24px;">
                Not because you have to. Because you want to. Because learning should feel like wonder, not work.
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 17px; color: #4b5563; line-height: 1.9; margin: 0 0 24px;">
                Every morning, I'll send you a 5-minute lesson. Some days it's science. Some days it's how to be a better friend. Always interesting. Always worth your time.
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.7; margin: 0 0 16px;">
                <strong>Today's lesson:</strong>
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 21px; color: #1f2937; line-height: 1.7; margin: 0 0 32px;">
                ${todayLesson.emoji} ${todayLesson.title}
              </p>
              
              <p style="margin: 0 0 32px;">
                <a href="https://curiouskelly.com/day/${dayNumber}" style="display: inline-block; background: #2563eb; color: white; padding: 14px 28px; border-radius: 8px; text-decoration: none; font-family: -apple-system, sans-serif; font-size: 15px; font-weight: 500;">
                  Start learning →
                </a>
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 15px; color: #6b7280; font-style: italic; margin: 0;">
                See you in your inbox tomorrow,<br>
                — Kelly
              </p>
            </td>
          </tr>
          
          <tr>
            <td style="padding: 24px; text-align: center; border-top: 1px solid #e5e7eb;">
              <p style="font-family: -apple-system, sans-serif; font-size: 12px; color: #9ca3af; margin: 0;">
                <a href="https://curiouskelly.com" style="color: #9ca3af;">curiouskelly.com</a>
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

function generateWelcomeEmailText(name: string, dayNumber: number, todayLesson: { title: string; emoji: string }): string {
  return `
Hi${name ? ' ' + name : ''},

I'm Kelly. I'm going to teach you one thing, every single day, for the rest of your life.

Not because you have to. Because you want to. Because learning should feel like wonder, not work.

Every morning, I'll send you a 5-minute lesson. Some days it's science. Some days it's how to be a better friend. Always interesting. Always worth your time.

Today's lesson: ${todayLesson.emoji} ${todayLesson.title}

Start learning: https://curiouskelly.com/day/${dayNumber}

See you in your inbox tomorrow,
— Kelly
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
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Verify webhook secret if configured
  if (webhookSecret) {
    const signature = req.headers['x-supabase-webhook-signature'];
    if (signature !== webhookSecret) {
      console.error('Invalid webhook signature');
      return res.status(401).json({ error: 'Unauthorized' });
    }
  }

  if (!supabaseUrl || !supabaseServiceKey || !resendApiKey) {
    console.error('Missing configuration');
    return res.status(500).json({ error: 'Server configuration error' });
  }

  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  try {
    const { type, record, old_record } = req.body;

    console.log('Auth webhook received:', type);

    // Handle user creation
    if (type === 'INSERT' && record?.email) {
      const email = record.email;
      const name = record.raw_user_meta_data?.name || record.raw_user_meta_data?.display_name || '';
      
      // Get today's lesson
      const dayOfYear = getDayOfYear();
      const { data: lesson } = await supabase
        .from('lessons')
        .select('title, emoji')
        .eq('day_number', dayOfYear)
        .single();

      const todayLesson = lesson || { title: 'Something wonderful', emoji: '✨' };

      // Send welcome email via Resend
      const response = await fetch(RESEND_API_URL, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${resendApiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          from: 'Kelly <hello@curiouskelly.com>',
          to: email,
          subject: 'Welcome to Curious Kelly',
          html: generateWelcomeEmailHTML(name, dayOfYear, todayLesson),
          text: generateWelcomeEmailText(name, dayOfYear, todayLesson),
          reply_to: 'hello@curiouskelly.com',
        }),
      });

      if (response.ok) {
        console.log(`Welcome email sent to ${email}`);
      } else {
        const errorData = await response.json();
        console.error('Failed to send welcome email:', errorData);
      }
    }

    return res.status(200).json({ received: true });

  } catch (error) {
    console.error('Webhook error:', error);
    return res.status(500).json({ error: 'Webhook processing failed' });
  }
}

