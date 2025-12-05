/**
 * Affiliate Email API
 * 
 * Sends affiliate-related emails:
 * - Welcome to affiliate program
 * - Payout notifications
 * 
 * Environment Variables:
 * - RESEND_API_KEY: Your Resend API key
 * - AFFILIATE_EMAIL_API_KEY: Secret key to authorize affiliate emails
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

const RESEND_API_URL = 'https://api.resend.com/emails';

const COLORS = {
  background: '#0a0a0b',
  cardBg: '#18181b',
  accent: '#3b82f6',
  gold: '#fbbf24',
  text: '#f4f4f5',
  textMuted: '#a1a1aa',
  textDim: '#71717a',
  border: '#27272a',
};

// Kelly's Voice: Humble, Curious, Collaborative, Warm, Simple, Rich
function generateAffiliateWelcomeHTML(name: string, affiliateCode: string, affiliateUrl: string): string {
  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; background-color: ${COLORS.background}; color: ${COLORS.text};">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: ${COLORS.background}; padding: 40px 20px;">
    <tr>
      <td align="center">
        <table width="100%" style="max-width: 500px;">
          <tr>
            <td style="padding: 40px 20px;">
              <p style="font-family: Georgia, serif; color: ${COLORS.text}; font-size: 19px; line-height: 1.9; margin: 0 0 24px;">
                ${name} —
              </p>
              
              <p style="font-family: Georgia, serif; color: ${COLORS.textMuted}; font-size: 17px; line-height: 1.9; margin: 0 0 24px;">
                Thank you for wanting to share this. It means a lot.
              </p>
              
              <p style="font-family: Georgia, serif; color: ${COLORS.textMuted}; font-size: 17px; line-height: 1.9; margin: 0 0 32px;">
                Here's your referral link — anyone who signs up through it gives you 30% of what they pay, for as long as they stay:
              </p>
              
              <!-- Link Box -->
              <table width="100%" style="background-color: ${COLORS.cardBg}; border-radius: 12px; margin: 0 0 32px;">
                <tr>
                  <td style="padding: 20px; text-align: center;">
                    <p style="font-family: monospace; color: ${COLORS.accent}; font-size: 14px; margin: 0; word-break: break-all;">
                      <a href="${affiliateUrl}" style="color: ${COLORS.accent}; text-decoration: none;">${affiliateUrl}</a>
                    </p>
                  </td>
                </tr>
              </table>
              
              <p style="font-family: Georgia, serif; color: ${COLORS.textMuted}; font-size: 17px; line-height: 1.9; margin: 0 0 32px;">
                That's it. No complicated dashboard. No targets. Just share when it feels right.
              </p>
              
              <p style="font-family: Georgia, serif; color: ${COLORS.textDim}; font-size: 15px; font-style: italic; margin: 0;">
                — Kelly
              </p>
              
              <p style="font-family: Georgia, serif; color: ${COLORS.textDim}; font-size: 14px; margin: 32px 0 0;">
                Questions? Just reply to this email.
              </p>
            </td>
          </tr>
          
          <!-- Footer -->
          <tr>
            <td style="padding: 24px 20px; text-align: center; border-top: 1px solid ${COLORS.border};">
              <p style="color: #52525b; font-size: 12px; margin: 0;">
                <a href="https://curiouskelly.com" style="color: #52525b;">curiouskelly.com</a>
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

function generateAffiliatePayoutHTML(name: string, amount: string, referralCount: number, payoutMethod: string, period: string): string {
  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; background-color: ${COLORS.background}; color: ${COLORS.text};">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: ${COLORS.background}; padding: 40px 20px;">
    <tr>
      <td align="center">
        <table width="100%" style="max-width: 500px;">
          <tr>
            <td style="padding: 40px 20px;">
              <p style="font-family: Georgia, serif; color: ${COLORS.text}; font-size: 19px; line-height: 1.9; margin: 0 0 24px;">
                ${name} —
              </p>
              
              <p style="font-family: Georgia, serif; color: ${COLORS.textMuted}; font-size: 17px; line-height: 1.9; margin: 0 0 24px;">
                ${referralCount} ${referralCount === 1 ? 'person' : 'people'} started learning because of you this month.
              </p>
              
              <p style="font-family: Georgia, serif; color: ${COLORS.textMuted}; font-size: 17px; line-height: 1.9; margin: 0 0 32px;">
                Your share is <strong style="color: ${COLORS.text};">${amount}</strong>, heading to your ${payoutMethod} in the next few days.
              </p>
              
              <p style="font-family: Georgia, serif; color: ${COLORS.textMuted}; font-size: 17px; line-height: 1.9; margin: 0 0 32px;">
                Thank you for helping more people find their curiosity.
              </p>
              
              <p style="font-family: Georgia, serif; color: ${COLORS.textDim}; font-size: 15px; font-style: italic; margin: 0;">
                — Kelly
              </p>
            </td>
          </tr>
          
          <!-- Footer -->
          <tr>
            <td style="padding: 24px 20px; text-align: center; border-top: 1px solid ${COLORS.border};">
              <p style="color: #52525b; font-size: 12px; margin: 0;">
                <a href="https://curiouskelly.com" style="color: #52525b;">curiouskelly.com</a>
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

interface AffiliateWelcomeRequest {
  type: 'welcome';
  email: string;
  name?: string;
  affiliateCode: string;
  affiliateUrl?: string;
}

interface AffiliatePayoutRequest {
  type: 'payout';
  email: string;
  name?: string;
  amount: string;
  referralCount: number;
  payoutMethod: string;
  period: string;
}

type AffiliateEmailRequest = AffiliateWelcomeRequest | AffiliatePayoutRequest;

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const authHeader = req.headers.authorization;
  const expectedKey = process.env.AFFILIATE_EMAIL_API_KEY;
  
  if (expectedKey && authHeader !== `Bearer ${expectedKey}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const resendApiKey = process.env.RESEND_API_KEY;
  if (!resendApiKey) {
    return res.status(500).json({ error: 'Email service not configured' });
  }

  try {
    const body = req.body as AffiliateEmailRequest;
    
    if (!body.email) {
      return res.status(400).json({ error: 'Email is required' });
    }

    if (!body.type) {
      return res.status(400).json({ error: 'Type is required', validTypes: ['welcome', 'payout'] });
    }

    let subject: string;
    let html: string;

    if (body.type === 'welcome') {
      const welcomeBody = body as AffiliateWelcomeRequest;
      if (!welcomeBody.affiliateCode) {
        return res.status(400).json({ error: 'affiliateCode is required for welcome emails' });
      }
      const affiliateUrl = welcomeBody.affiliateUrl || `https://curiouskelly.com/?ref=${welcomeBody.affiliateCode}`;
      subject = '🤝 Welcome to the Curious Kelly Affiliate Program!';
      html = generateAffiliateWelcomeHTML(welcomeBody.name || 'Partner', welcomeBody.affiliateCode, affiliateUrl);
    } else if (body.type === 'payout') {
      const payoutBody = body as AffiliatePayoutRequest;
      if (!payoutBody.amount || !payoutBody.referralCount || !payoutBody.payoutMethod || !payoutBody.period) {
        return res.status(400).json({ 
          error: 'Missing required fields for payout email',
          required: ['amount', 'referralCount', 'payoutMethod', 'period']
        });
      }
      subject = `💰 Your ${payoutBody.amount} affiliate payout is on the way!`;
      html = generateAffiliatePayoutHTML(payoutBody.name || 'Partner', payoutBody.amount, payoutBody.referralCount, payoutBody.payoutMethod, payoutBody.period);
    } else {
      return res.status(400).json({ error: 'Invalid email type', validTypes: ['welcome', 'payout'] });
    }

    const response = await fetch(RESEND_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${resendApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'Kelly <hello@curiouskelly.com>',
        to: body.email,
        subject,
        html,
        reply_to: 'hello@curiouskelly.com',
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      return res.status(500).json({ error: 'Failed to send email', details: data });
    }

    return res.status(200).json({ 
      success: true,
      message: `Affiliate ${body.type} email sent`,
      id: data.id,
    });

  } catch (error) {
    console.error('Error sending affiliate email:', error);
    return res.status(500).json({ 
      error: 'Failed to send email',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}
