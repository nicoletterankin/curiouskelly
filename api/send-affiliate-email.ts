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

function generateAffiliateWelcomeHTML(name: string, affiliateCode: string, affiliateUrl: string): string {
  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: ${COLORS.background}; color: ${COLORS.text};">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: ${COLORS.background}; padding: 40px 20px;">
    <tr>
      <td align="center">
        <table width="100%" style="max-width: 560px; background-color: ${COLORS.cardBg}; border-radius: 16px; overflow: hidden;">
          <tr>
            <td style="padding: 40px 40px 20px; text-align: center;">
              <img src="https://curiouskelly.com/images/brand/kelly-mark-circle-64.png" alt="Kelly" width="80" height="80" style="border-radius: 50%; border: 3px solid ${COLORS.accent};">
              <h1 style="color: ${COLORS.text}; font-size: 28px; font-weight: 600; margin: 20px 0 0;">Welcome to the Kelly Affiliate Family! 🤝</h1>
            </td>
          </tr>
          <tr>
            <td style="padding: 0 40px 30px;">
              <p style="color: ${COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">Hey ${name}!</p>
              <p style="color: ${COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">I'm thrilled to have you as a Curious Kelly partner! Together, we're going to help more people discover the joy of learning something new every day.</p>
              
              <!-- Affiliate Code Box -->
              <table width="100%" style="background-color: ${COLORS.border}; border-radius: 12px; margin: 30px 0;">
                <tr>
                  <td style="padding: 24px; text-align: center;">
                    <p style="color: ${COLORS.textDim}; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 12px;">Your Affiliate Code</p>
                    <p style="color: ${COLORS.gold}; font-size: 32px; font-weight: 700; font-family: monospace; margin: 0 0 16px;">${affiliateCode}</p>
                    <p style="color: ${COLORS.textMuted}; font-size: 14px; margin: 0;">Your referral link:<br><a href="${affiliateUrl}" style="color: ${COLORS.accent}; word-break: break-all;">${affiliateUrl}</a></p>
                  </td>
                </tr>
              </table>
              
              <p style="color: ${COLORS.text}; font-size: 16px; line-height: 1.6; margin: 0 0 10px; font-weight: 600;">Here's what you earn:</p>
              <ul style="color: ${COLORS.textMuted}; font-size: 16px; line-height: 1.8; margin: 0 0 20px; padding-left: 20px;">
                <li><strong style="color: ${COLORS.gold};">30% commission</strong> on every subscription</li>
                <li>Recurring revenue for as long as they stay subscribed</li>
                <li>Payouts every month via PayPal or bank transfer</li>
                <li>Real-time dashboard to track your referrals</li>
              </ul>
              
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td align="center" style="padding: 20px 0;">
                    <a href="https://curiouskelly.com/affiliate/dashboard" style="display: inline-block; background-color: ${COLORS.accent}; color: white; text-decoration: none; padding: 14px 32px; border-radius: 8px; font-weight: 600; font-size: 16px;">Access Your Dashboard →</a>
                  </td>
                </tr>
              </table>
              
              <p style="color: ${COLORS.text}; font-size: 16px; line-height: 1.6; margin: 20px 0 0;">✨ Let's grow together,<br><strong>Kelly</strong></p>
              <p style="color: ${COLORS.textDim}; font-size: 14px; margin: 20px 0 0;">Questions? Reply to this email anytime!</p>
            </td>
          </tr>
          <tr>
            <td style="padding: 30px 40px; border-top: 1px solid ${COLORS.border}; text-align: center;">
              <p style="color: #52525b; font-size: 12px; margin: 0 0 10px;">✨ Curious Kelly | Learn something new every day</p>
              <p style="color: #52525b; font-size: 12px; margin: 0;">
                <a href="https://curiouskelly.com" style="color: #52525b;">curiouskelly.com</a> · 
                <a href="https://curiouskelly.com/help" style="color: #52525b;">Help</a>
              </p>
              <p style="color: #3f3f46; font-size: 11px; margin: 15px 0 0;">© 2025 Lesson of the Day PBC</p>
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
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: ${COLORS.background}; color: ${COLORS.text};">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: ${COLORS.background}; padding: 40px 20px;">
    <tr>
      <td align="center">
        <table width="100%" style="max-width: 560px; background-color: ${COLORS.cardBg}; border-radius: 16px; overflow: hidden;">
          <tr>
            <td style="padding: 40px 40px 20px; text-align: center;">
              <img src="https://curiouskelly.com/images/brand/kelly-mark-circle-64.png" alt="Kelly" width="80" height="80" style="border-radius: 50%; border: 3px solid ${COLORS.accent};">
            </td>
          </tr>
          <tr>
            <td style="padding: 0 40px 30px; text-align: center;">
              <p style="font-size: 48px; margin: 0 0 16px;">💰</p>
              <h1 style="color: ${COLORS.text}; font-size: 28px; font-weight: 600; margin: 0 0 8px;">Payout on the way!</h1>
              <p style="color: ${COLORS.textMuted}; font-size: 16px; margin: 0 0 30px;">Great work, ${name}!</p>
              
              <!-- Payout Details -->
              <table width="100%" style="background-color: ${COLORS.border}; border-radius: 12px; margin: 0 0 30px;">
                <tr>
                  <td style="padding: 24px; text-align: center;">
                    <p style="color: ${COLORS.gold}; font-size: 48px; font-weight: 700; margin: 0 0 8px;">${amount}</p>
                    <p style="color: ${COLORS.textMuted}; font-size: 14px; margin: 0;">From ${referralCount} referral${referralCount !== 1 ? 's' : ''} · ${period}</p>
                  </td>
                </tr>
              </table>
              
              <p style="color: ${COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">Your payout is being sent via <strong style="color: ${COLORS.text};">${payoutMethod}</strong> and should arrive within 2-3 business days.</p>
              
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td align="center" style="padding: 20px 0;">
                    <a href="https://curiouskelly.com/affiliate/payouts" style="display: inline-block; background-color: ${COLORS.accent}; color: white; text-decoration: none; padding: 14px 32px; border-radius: 8px; font-weight: 600; font-size: 16px;">View Payout Details →</a>
                  </td>
                </tr>
              </table>
              
              <p style="color: ${COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 30px 0 0;">Thank you for spreading curiosity! 🙏</p>
              <p style="color: ${COLORS.text}; font-size: 16px; line-height: 1.6; margin: 20px 0 0;">✨ Gratefully,<br><strong>Kelly</strong></p>
            </td>
          </tr>
          <tr>
            <td style="padding: 30px 40px; border-top: 1px solid ${COLORS.border}; text-align: center;">
              <p style="color: #52525b; font-size: 12px; margin: 0 0 10px;">✨ Curious Kelly | Learn something new every day</p>
              <p style="color: #52525b; font-size: 12px; margin: 0;">
                <a href="https://curiouskelly.com" style="color: #52525b;">curiouskelly.com</a> · 
                <a href="https://curiouskelly.com/help" style="color: #52525b;">Help</a>
              </p>
              <p style="color: #3f3f46; font-size: 11px; margin: 15px 0 0;">© 2025 Lesson of the Day PBC</p>
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
