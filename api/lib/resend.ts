/**
 * Resend Email Service
 * 
 * Centralized email sending utility for all Curious Kelly emails.
 * Uses Resend API for reliable delivery.
 * 
 * @module resend
 */

const RESEND_API_URL = 'https://api.resend.com/emails';
const RESEND_BATCH_URL = 'https://api.resend.com/emails/batch';

export interface EmailOptions {
  to: string | string[];
  subject: string;
  html: string;
  text?: string;
  from?: string;
  replyTo?: string;
  tags?: Array<{ name: string; value: string }>;
  scheduledAt?: string; // ISO 8601 format for scheduled emails
}

export interface EmailResult {
  success: boolean;
  id?: string;
  error?: string;
  details?: unknown;
}

export interface BatchEmailResult {
  success: boolean;
  data?: Array<{ id: string }>;
  error?: string;
  details?: unknown;
}

/**
 * Send a single email via Resend
 */
export async function sendEmail(options: EmailOptions): Promise<EmailResult> {
  const apiKey = process.env.RESEND_API_KEY;
  
  if (!apiKey) {
    console.error('RESEND_API_KEY not configured');
    return { success: false, error: 'Email service not configured' };
  }

  try {
    const response = await fetch(RESEND_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: options.from || 'Kelly <hello@curiouskelly.com>',
        to: options.to,
        subject: options.subject,
        html: options.html,
        text: options.text,
        reply_to: options.replyTo || 'hello@curiouskelly.com',
        tags: options.tags,
        scheduled_at: options.scheduledAt,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      console.error('Resend API error:', data);
      return { success: false, error: 'Failed to send email', details: data };
    }

    console.log(`Email sent successfully:`, data);
    return { success: true, id: data.id };

  } catch (error) {
    console.error('Error sending email:', error);
    return { 
      success: false, 
      error: 'Failed to send email',
      details: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

/**
 * Send multiple emails in a batch (up to 100 at a time)
 * Great for daily lesson emails!
 */
export async function sendBatchEmails(emails: EmailOptions[]): Promise<BatchEmailResult> {
  const apiKey = process.env.RESEND_API_KEY;
  
  if (!apiKey) {
    console.error('RESEND_API_KEY not configured');
    return { success: false, error: 'Email service not configured' };
  }

  // Resend allows max 100 emails per batch
  if (emails.length > 100) {
    console.error('Batch size exceeds 100 emails');
    return { success: false, error: 'Batch size exceeds maximum of 100' };
  }

  try {
    const response = await fetch(RESEND_BATCH_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(
        emails.map(email => ({
          from: email.from || 'Kelly <hello@curiouskelly.com>',
          to: email.to,
          subject: email.subject,
          html: email.html,
          text: email.text,
          reply_to: email.replyTo || 'hello@curiouskelly.com',
          tags: email.tags,
        }))
      ),
    });

    const data = await response.json();

    if (!response.ok) {
      console.error('Resend batch API error:', data);
      return { success: false, error: 'Failed to send batch emails', details: data };
    }

    console.log(`Batch emails sent successfully: ${emails.length} emails`, data);
    return { success: true, data: data.data };

  } catch (error) {
    console.error('Error sending batch emails:', error);
    return { 
      success: false, 
      error: 'Failed to send batch emails',
      details: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

/**
 * Schedule an email for future delivery
 */
export async function scheduleEmail(
  options: Omit<EmailOptions, 'scheduledAt'>,
  sendAt: Date
): Promise<EmailResult> {
  return sendEmail({
    ...options,
    scheduledAt: sendAt.toISOString(),
  });
}

/**
 * Email type tags for analytics and filtering
 */
export const EMAIL_TAGS = {
  WELCOME: { name: 'type', value: 'welcome' },
  DAILY_LESSON: { name: 'type', value: 'daily_lesson' },
  STREAK: { name: 'type', value: 'streak' },
  AFFILIATE_WELCOME: { name: 'type', value: 'affiliate_welcome' },
  AFFILIATE_PAYOUT: { name: 'type', value: 'affiliate_payout' },
  RE_ENGAGEMENT: { name: 'type', value: 're_engagement' },
  PASSWORD_RESET: { name: 'type', value: 'password_reset' },
  SUBSCRIPTION: { name: 'type', value: 'subscription' },
  GIFT_RECEIVED: { name: 'type', value: 'gift_received' },
  GIFT_REDEEMED: { name: 'type', value: 'gift_redeemed' },
} as const;

