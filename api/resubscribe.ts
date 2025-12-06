/**
 * Resubscribe Endpoint
 * 
 * Re-enables email subscriptions for users who previously unsubscribed.
 * 
 * GET /api/resubscribe?token=UUID
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

function generateResubscribePage(success: boolean, message: string): string {
  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${success ? 'Welcome Back' : 'Error'} - Curious Kelly</title>
  <link rel="icon" href="/favicon.ico">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: Georgia, serif;
      background: #fafafa;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }
    .container {
      max-width: 460px;
      text-align: center;
    }
    .icon {
      font-size: 48px;
      margin-bottom: 24px;
    }
    h1 {
      font-size: 24px;
      color: #1f2937;
      font-weight: 500;
      margin-bottom: 16px;
    }
    p {
      font-size: 17px;
      color: #4b5563;
      line-height: 1.8;
      margin-bottom: 24px;
    }
    .signature {
      color: #6b7280;
      font-style: italic;
      font-size: 15px;
    }
    a.button {
      display: inline-block;
      background: #2563eb;
      color: white;
      padding: 12px 24px;
      border-radius: 8px;
      text-decoration: none;
      font-family: -apple-system, sans-serif;
      font-size: 14px;
      margin-top: 16px;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="icon">${success ? '✨' : '😔'}</div>
    <h1>${success ? 'Welcome back' : 'Something went wrong'}</h1>
    <p>${message}</p>
    ${success ? '<p class="signature">— Kelly</p>' : ''}
    <a href="${success ? 'https://curiouskelly.com/learn' : 'https://curiouskelly.com/help'}" class="button">
      ${success ? 'Start Today\'s Lesson' : 'Get Help'}
    </a>
  </div>
</body>
</html>
  `.trim();
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  if (!supabaseUrl || !supabaseServiceKey) {
    return res.status(500).send(generateResubscribePage(
      false,
      'Could not update your preferences. Please contact hello@curiouskelly.com'
    ));
  }

  const supabase = createClient(supabaseUrl, supabaseServiceKey);
  const token = req.query.token as string;

  if (!token) {
    return res.status(400).send(generateResubscribePage(
      false,
      'Invalid link. Please contact hello@curiouskelly.com for help.'
    ));
  }

  try {
    // Find user by token
    const { data: user, error: findError } = await supabase
      .from('users')
      .select('id, email')
      .eq('unsubscribe_token', token)
      .single();

    if (findError || !user) {
      return res.status(404).send(generateResubscribePage(
        false,
        'Could not find your account. This link may have expired.'
      ));
    }

    // Re-enable emails
    const { error: updateError } = await supabase
      .from('users')
      .update({
        email_daily_lesson: true,
        email_streak_notifications: true,
        email_unsubscribed_at: null
      })
      .eq('id', user.id);

    if (updateError) {
      return res.status(500).send(generateResubscribePage(
        false,
        'Could not update your preferences. Please try again.'
      ));
    }

    console.log(`Resubscribed user ${user.email}`);

    return res.status(200).send(generateResubscribePage(
      true,
      'I\'m so glad you\'re back. Tomorrow morning, you\'ll find a new lesson in your inbox. I can\'t wait to learn together again.'
    ));

  } catch (error) {
    console.error('Resubscribe error:', error);
    return res.status(500).send(generateResubscribePage(
      false,
      'Something unexpected happened. Please contact hello@curiouskelly.com'
    ));
  }
}

