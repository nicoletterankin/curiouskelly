/**
 * Supabase Auth Webhook Handler
 * 
 * Receives webhooks from Supabase when auth events occur.
 * Automatically sends welcome emails to new users.
 * 
 * Setup in Supabase:
 * 1. Go to Database → Webhooks
 * 2. Create new webhook:
 *    - Name: send-welcome-email
 *    - Table: auth.users
 *    - Events: INSERT
 *    - URL: https://www.curiouskelly.com/api/supabase-auth-webhook
 *    - HTTP Headers: 
 *      - x-webhook-secret: (your secret from SUPABASE_WEBHOOK_SECRET env var)
 * 
 * Environment Variables:
 * - SUPABASE_WEBHOOK_SECRET: Secret to verify webhook authenticity
 * - RESEND_API_KEY: Your Resend API key
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

const RESEND_API_URL = 'https://api.resend.com/emails';

const FUN_FACTS = [
  "The average person stops actively learning around age 25. You just changed that!",
  "You create 700 new neural connections every time you learn something new.",
  "Children ask ~300 questions per day. Adults? About 20. Let's change that!",
  "Reading 20 minutes a day exposes you to 1.8 million words per year.",
  "Einstein failed his university entrance exam the first time. Look where curiosity took him!",
  "Your brain uses 20% of your body's energy but is only 2% of your weight.",
];

function getRandomFact(): string {
  return FUN_FACTS[Math.floor(Math.random() * FUN_FACTS.length)];
}

function generateWelcomeEmailHTML(name: string): string {
  // Kelly's Voice: Humble, Curious, Collaborative, Warm, Simple, Rich
  return `
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

Hi — I'm Kelly.<br><br>

I don't have all the answers. But I love finding them. And I think learning is better together.<br><br>

Every day I find something wonderful and I can't wait to share it. Today's lesson is ready.<br><br>

<a href="https://curiouskelly.com/learn" style="color: #1e3a5f; text-decoration: underline;">Want to come along?</a><br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
  `.trim();
}

function generateWelcomeEmailText(name: string): string {
  // Kelly's Voice: Humble, Curious, Collaborative, Warm, Simple, Rich
  return `
Hi — I'm Kelly.

I don't have all the answers. But I love finding them. And I think learning is better together.

Every day I find something wonderful and I can't wait to share it. Today's lesson is ready.

Want to come along? https://curiouskelly.com/learn

— Kelly

---
✨ Curious Kelly | Learn something new every day
curiouskelly.com | © 2025 Lesson of the Day PBC
  `.trim();
}

interface SupabaseWebhookPayload {
  type: 'INSERT' | 'UPDATE' | 'DELETE';
  table: string;
  schema: string;
  record: {
    id: string;
    email: string;
    raw_user_meta_data?: {
      name?: string;
      full_name?: string;
    };
  };
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const webhookSecret = process.env.SUPABASE_WEBHOOK_SECRET;
  const receivedSecret = req.headers['x-webhook-secret'];
  
  if (webhookSecret && receivedSecret !== webhookSecret) {
    console.error('Invalid webhook secret');
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const resendApiKey = process.env.RESEND_API_KEY;
  if (!resendApiKey) {
    return res.status(500).json({ error: 'Email service not configured' });
  }

  try {
    const payload = req.body as SupabaseWebhookPayload;
    
    console.log('Received Supabase webhook:', {
      type: payload.type,
      table: payload.table,
      schema: payload.schema,
      userId: payload.record?.id,
    });

    if (payload.type !== 'INSERT') {
      return res.status(200).json({ message: 'Ignored non-INSERT event' });
    }

    if (payload.table !== 'users' || payload.schema !== 'auth') {
      return res.status(200).json({ message: 'Ignored non-auth.users event' });
    }

    const { record } = payload;
    
    if (!record.email) {
      return res.status(400).json({ error: 'No email in user record' });
    }

    const name = record.raw_user_meta_data?.name 
      || record.raw_user_meta_data?.full_name
      || record.email.split('@')[0];

    const response = await fetch(RESEND_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${resendApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'Kelly <hello@curiouskelly.com>',
        to: record.email,
        subject: `Welcome to curiosity, ${name}! 🎉`,
        html: generateWelcomeEmailHTML(name),
        text: generateWelcomeEmailText(name),
        reply_to: 'hello@curiouskelly.com',
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      console.error('Resend API error:', data);
      return res.status(500).json({ error: 'Failed to send welcome email', details: data });
    }

    console.log(`Welcome email sent to ${record.email}`, { emailId: data.id });
    
    return res.status(200).json({ 
      success: true,
      message: 'Welcome email sent',
      emailId: data.id,
      userId: record.id,
    });

  } catch (error) {
    console.error('Error processing webhook:', error);
    return res.status(500).json({ 
      error: 'Failed to process webhook',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}
