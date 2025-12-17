/**
 * Email Sending Utilities
 * 
 * Uses Resend API for transactional emails.
 * Fallback: logs to console if RESEND_API_KEY not configured.
 */

interface GiftEmailData {
  recipientEmail: string;
  giftCode: string;
  gifterName?: string;
  giftMessage?: string;
  durationMonths: number; // 0 = lifetime
  purchaseAmount: number;
}

interface RenewalReminderData {
  customerEmail: string;
  daysUntilDue: number;
  amountDue: number; // in cents
  planType?: string;
}

interface EmailResult {
  success: boolean;
  messageId?: string;
  error?: string;
}

const SENDER_EMAIL = 'hello@curiouskelly.com';
const SENDER_NAME = 'Curious Kelly';

/**
 * Send a gift code email to the recipient
 */
export async function sendGiftEmail(data: GiftEmailData): Promise<EmailResult> {
  const resendKey = process.env.RESEND_API_KEY;
  
  if (!resendKey) {
    console.log('[Email] RESEND_API_KEY not configured, logging gift email instead:');
    console.log('[Email] Gift Code Email:', JSON.stringify(data, null, 2));
    return { success: true, messageId: 'logged-only' };
  }

  const durationText = data.durationMonths === 0 
    ? 'Lifetime Access' 
    : `${data.durationMonths} Month${data.durationMonths > 1 ? 's' : ''}`;

  const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>You've Received a Gift!</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f4f4f5;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #f4f4f5;">
    <tr>
      <td align="center" style="padding: 40px 20px;">
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="max-width: 500px; background-color: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
          <!-- Header -->
          <tr>
            <td style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); padding: 40px 30px; text-align: center;">
              <div style="font-size: 48px; margin-bottom: 16px;">🎁</div>
              <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 600;">You've Received a Gift!</h1>
            </td>
          </tr>
          
          <!-- Body -->
          <tr>
            <td style="padding: 40px 30px;">
              ${data.gifterName ? `<p style="margin: 0 0 20px 0; font-size: 16px; color: #374151;">${data.gifterName} has sent you a special gift:</p>` : ''}
              
              <div style="background-color: #f0f9ff; border: 2px solid #3b82f6; border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 24px;">
                <p style="margin: 0 0 8px 0; font-size: 14px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px;">Your Gift</p>
                <p style="margin: 0 0 16px 0; font-size: 24px; font-weight: 600; color: #1e40af;">✨ Curious Kelly ${durationText}</p>
                <p style="margin: 0; font-size: 14px; color: #6b7280;">365 days of curiosity, wonder, and learning</p>
              </div>
              
              ${data.giftMessage ? `
              <div style="background-color: #fefce8; border-left: 4px solid #fbbf24; padding: 16px; margin-bottom: 24px; border-radius: 0 8px 8px 0;">
                <p style="margin: 0 0 8px 0; font-size: 12px; color: #92400e; text-transform: uppercase; letter-spacing: 1px;">Personal Message</p>
                <p style="margin: 0; font-size: 16px; color: #78350f; font-style: italic;">"${data.giftMessage}"</p>
              </div>
              ` : ''}
              
              <!-- Gift Code -->
              <div style="text-align: center; margin-bottom: 32px;">
                <p style="margin: 0 0 12px 0; font-size: 14px; color: #6b7280;">Your Gift Code:</p>
                <div style="background-color: #18181b; color: #fbbf24; font-size: 28px; font-weight: 700; letter-spacing: 4px; padding: 16px 24px; border-radius: 8px; display: inline-block; font-family: monospace;">
                  ${data.giftCode}
                </div>
              </div>
              
              <!-- CTA Button -->
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                <tr>
                  <td align="center">
                    <a href="https://curiouskelly.com/redeem?code=${data.giftCode}" 
                       style="display: inline-block; background-color: #3b82f6; color: #ffffff; font-size: 16px; font-weight: 600; text-decoration: none; padding: 16px 32px; border-radius: 8px;">
                      Redeem Your Gift →
                    </a>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          
          <!-- Footer -->
          <tr>
            <td style="background-color: #f9fafb; padding: 24px 30px; text-align: center; border-top: 1px solid #e5e7eb;">
              <p style="margin: 0 0 8px 0; font-size: 14px; color: #6b7280;">
                Questions? Email us at <a href="mailto:hello@curiouskelly.com" style="color: #3b82f6;">hello@curiouskelly.com</a>
              </p>
              <p style="margin: 0; font-size: 12px; color: #9ca3af;">
                © ${new Date().getFullYear()} Lesson of the Day PBC. All rights reserved.
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

  try {
    const response = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${resendKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: `${SENDER_NAME} <${SENDER_EMAIL}>`,
        to: [data.recipientEmail],
        subject: `🎁 ${data.gifterName || 'Someone'} sent you a gift: Curious Kelly!`,
        html,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('[Email] Resend API error:', errorText);
      return { success: false, error: errorText };
    }

    const result = await response.json();
    console.log('[Email] Gift email sent:', result.id);
    return { success: true, messageId: result.id };

  } catch (error) {
    console.error('[Email] Failed to send gift email:', error);
    return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
}

/**
 * Send renewal reminder email
 */
export async function sendRenewalReminderEmail(data: RenewalReminderData): Promise<EmailResult> {
  const resendKey = process.env.RESEND_API_KEY;
  
  if (!resendKey) {
    console.log('[Email] RESEND_API_KEY not configured, logging renewal reminder instead:');
    console.log('[Email] Renewal Reminder:', JSON.stringify(data, null, 2));
    return { success: true, messageId: 'logged-only' };
  }

  const amountFormatted = (data.amountDue / 100).toFixed(2);

  const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f4f4f5;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #f4f4f5;">
    <tr>
      <td align="center" style="padding: 40px 20px;">
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="max-width: 500px; background-color: #ffffff; border-radius: 16px; overflow: hidden;">
          <tr>
            <td style="padding: 40px 30px; text-align: center;">
              <div style="font-size: 48px; margin-bottom: 16px;">📅</div>
              <h1 style="margin: 0 0 16px 0; color: #18181b; font-size: 24px;">Your Subscription Renews Soon</h1>
              <p style="margin: 0 0 24px 0; color: #6b7280; font-size: 16px;">
                Just a heads up — your Curious Kelly subscription will renew in <strong>${data.daysUntilDue} days</strong> for <strong>$${amountFormatted}</strong>.
              </p>
              <p style="margin: 0 0 24px 0; color: #6b7280; font-size: 14px;">
                No action needed if you want to continue learning. We'll charge the payment method on file.
              </p>
              <a href="https://curiouskelly.com/account" 
                 style="display: inline-block; background-color: #3b82f6; color: #ffffff; font-size: 14px; font-weight: 600; text-decoration: none; padding: 12px 24px; border-radius: 8px;">
                Manage Subscription
              </a>
            </td>
          </tr>
          <tr>
            <td style="background-color: #f9fafb; padding: 20px 30px; text-align: center; border-top: 1px solid #e5e7eb;">
              <p style="margin: 0; font-size: 12px; color: #9ca3af;">
                © ${new Date().getFullYear()} Lesson of the Day PBC
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

  try {
    const response = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${resendKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: `${SENDER_NAME} <${SENDER_EMAIL}>`,
        to: [data.customerEmail],
        subject: `📅 Your Curious Kelly subscription renews in ${data.daysUntilDue} days`,
        html,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('[Email] Resend API error:', errorText);
      return { success: false, error: errorText };
    }

    const result = await response.json();
    console.log('[Email] Renewal reminder sent:', result.id);
    return { success: true, messageId: result.id };

  } catch (error) {
    console.error('[Email] Failed to send renewal reminder:', error);
    return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
}
