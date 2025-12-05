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
import { sendEmail, EMAIL_TAGS } from './lib/resend';
import { welcomeEmail } from './lib/email-templates';

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
      avatar_url?: string;
    };
    created_at: string;
    updated_at: string;
    email_confirmed_at?: string;
  };
  old_record?: unknown;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Verify webhook secret
  const webhookSecret = process.env.SUPABASE_WEBHOOK_SECRET;
  const receivedSecret = req.headers['x-webhook-secret'];
  
  if (webhookSecret && receivedSecret !== webhookSecret) {
    console.error('Invalid webhook secret');
    return res.status(401).json({ error: 'Unauthorized' });
  }

  try {
    const payload = req.body as SupabaseWebhookPayload;
    
    console.log('Received Supabase webhook:', {
      type: payload.type,
      table: payload.table,
      schema: payload.schema,
      userId: payload.record?.id,
    });

    // Only process INSERT events on auth.users
    if (payload.type !== 'INSERT') {
      return res.status(200).json({ message: 'Ignored non-INSERT event' });
    }

    if (payload.table !== 'users' || payload.schema !== 'auth') {
      return res.status(200).json({ message: 'Ignored non-auth.users event' });
    }

    const { record } = payload;
    
    if (!record.email) {
      console.error('No email in user record');
      return res.status(400).json({ error: 'No email in user record' });
    }

    // Extract name from user metadata
    const name = record.raw_user_meta_data?.name 
      || record.raw_user_meta_data?.full_name
      || record.email.split('@')[0]; // Fallback to email prefix

    // Generate welcome email
    const email = welcomeEmail(name);
    
    // Send the email
    const result = await sendEmail({
      to: record.email,
      subject: email.subject,
      html: email.html,
      text: email.text,
      tags: [EMAIL_TAGS.WELCOME],
    });

    if (!result.success) {
      console.error('Failed to send welcome email:', result.error);
      return res.status(500).json({ 
        error: 'Failed to send welcome email',
        details: result.details 
      });
    }

    console.log(`Welcome email sent to ${record.email}`, { emailId: result.id });
    
    return res.status(200).json({ 
      success: true,
      message: 'Welcome email sent',
      emailId: result.id,
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

