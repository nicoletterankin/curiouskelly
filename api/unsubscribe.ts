/**
 * Unsubscribe Endpoint
 * 
 * Handles one-click unsubscribe from daily lesson emails.
 * Supports both GET (one-click from email) and POST (from web form).
 * 
 * GET /api/unsubscribe?token=UUID
 * POST /api/unsubscribe { token: UUID, type?: 'daily' | 'streak' | 'all' }
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

/**
 * Generate unsubscribe confirmation HTML page
 */
function generateUnsubscribePage(success: boolean, message: string, resubscribeUrl?: string): string {
  // Kelly's Voice: Humble, Curious, Collaborative, Warm, Simple, Rich
  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${success ? 'Unsubscribed' : 'Error'} - Curious Kelly</title>
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
    a.link {
      color: #2563eb;
      text-decoration: underline;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="icon">${success ? '👋' : '😔'}</div>
    <h1>${success ? 'You\'re unsubscribed' : 'Something went wrong'}</h1>
    <p>${message}</p>
    ${success ? `
      <p class="signature">— Kelly</p>
      ${resubscribeUrl ? `
        <p style="margin-top: 32px; font-size: 14px; color: #9ca3af;">
          Changed your mind? <a href="${resubscribeUrl}" class="link">Resubscribe</a>
        </p>
      ` : ''}
      <a href="https://curiouskelly.com" class="button">Back to Curious Kelly</a>
    ` : `
      <a href="https://curiouskelly.com/help" class="button">Get Help</a>
    `}
  </div>
</body>
</html>
  `.trim();
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (!supabaseUrl || !supabaseServiceKey) {
    return res.status(500).send(generateUnsubscribePage(
      false,
      'Email preferences could not be updated. Please try again later or contact hello@curiouskelly.com'
    ));
  }

  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  // Get token from query (GET) or body (POST)
  const token = req.method === 'GET' 
    ? req.query.token as string
    : req.body?.token;

  // Get unsubscribe type (default: daily)
  const type = req.method === 'POST' 
    ? req.body?.type || 'daily'
    : 'daily';

  if (!token) {
    return res.status(400).send(generateUnsubscribePage(
      false,
      'Invalid unsubscribe link. If you copied this from an email, make sure you got the full link.'
    ));
  }

  try {
    // Find user by unsubscribe token
    const { data: user, error: findError } = await supabase
      .from('users')
      .select('id, email')
      .eq('unsubscribe_token', token)
      .single();

    if (findError || !user) {
      console.error('Unsubscribe: User not found for token', token);
      return res.status(404).send(generateUnsubscribePage(
        false,
        'We couldn\'t find your subscription. This link may have expired. Please contact hello@curiouskelly.com if you need help.'
      ));
    }

    // Update based on type
    let updateData: Record<string, any> = {};
    let message: string;

    switch (type) {
      case 'all':
        updateData = {
          email_daily_lesson: false,
          email_streak_notifications: false,
          email_unsubscribed_at: new Date().toISOString()
        };
        message = 'You won\'t receive any more emails from me. I understand — everyone\'s inbox is precious. If you ever want to learn together again, I\'ll be here.';
        break;
      
      case 'streak':
        updateData = { email_streak_notifications: false };
        message = 'You won\'t receive streak celebration emails anymore. You\'ll still get daily lessons if you\'re subscribed.';
        break;
      
      case 'daily':
      default:
        updateData = { email_daily_lesson: false };
        message = 'You won\'t receive daily lesson emails anymore. I\'ll miss you in my inbox, but I understand. The lessons will always be here if you want to come back.';
        break;
    }

    const { error: updateError } = await supabase
      .from('users')
      .update(updateData)
      .eq('id', user.id);

    if (updateError) {
      console.error('Unsubscribe: Update failed', updateError);
      return res.status(500).send(generateUnsubscribePage(
        false,
        'Something went wrong updating your preferences. Please try again or contact hello@curiouskelly.com'
      ));
    }

    console.log(`Unsubscribed user ${user.email} from ${type} emails`);

    // Return success page
    const resubscribeUrl = `https://curiouskelly.com/api/resubscribe?token=${token}`;
    
    return res.status(200).send(generateUnsubscribePage(
      true,
      message,
      resubscribeUrl
    ));

  } catch (error) {
    console.error('Unsubscribe error:', error);
    return res.status(500).send(generateUnsubscribePage(
      false,
      'Something unexpected happened. Please try again or contact hello@curiouskelly.com'
    ));
  }
}


