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
import { sendEmail, EMAIL_TAGS } from './lib/resend';
import { affiliateWelcomeEmail, affiliatePayoutEmail } from './lib/email-templates';

type EmailType = 'welcome' | 'payout';

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
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Verify API key
  const authHeader = req.headers.authorization;
  const expectedKey = process.env.AFFILIATE_EMAIL_API_KEY;
  
  if (expectedKey && authHeader !== `Bearer ${expectedKey}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  try {
    const body = req.body as AffiliateEmailRequest;
    
    if (!body.email) {
      return res.status(400).json({ error: 'Email is required' });
    }

    if (!body.type) {
      return res.status(400).json({ 
        error: 'Type is required',
        validTypes: ['welcome', 'payout']
      });
    }

    let emailContent: { subject: string; html: string; text: string };
    let tags: Array<{ name: string; value: string }>;

    switch (body.type) {
      case 'welcome': {
        const welcomeBody = body as AffiliateWelcomeRequest;
        
        if (!welcomeBody.affiliateCode) {
          return res.status(400).json({ error: 'affiliateCode is required for welcome emails' });
        }
        
        const affiliateUrl = welcomeBody.affiliateUrl 
          || `https://curiouskelly.com/?ref=${welcomeBody.affiliateCode}`;
        
        emailContent = affiliateWelcomeEmail(
          welcomeBody.name || 'Partner',
          welcomeBody.affiliateCode,
          affiliateUrl
        );
        tags = [EMAIL_TAGS.AFFILIATE_WELCOME];
        break;
      }

      case 'payout': {
        const payoutBody = body as AffiliatePayoutRequest;
        
        if (!payoutBody.amount || !payoutBody.referralCount || !payoutBody.payoutMethod || !payoutBody.period) {
          return res.status(400).json({ 
            error: 'Missing required fields for payout email',
            required: ['amount', 'referralCount', 'payoutMethod', 'period']
          });
        }
        
        emailContent = affiliatePayoutEmail(
          payoutBody.name || 'Partner',
          payoutBody.amount,
          payoutBody.referralCount,
          payoutBody.payoutMethod,
          payoutBody.period
        );
        tags = [EMAIL_TAGS.AFFILIATE_PAYOUT];
        break;
      }

      default:
        return res.status(400).json({ 
          error: 'Invalid email type',
          validTypes: ['welcome', 'payout']
        });
    }

    const result = await sendEmail({
      to: body.email,
      subject: emailContent.subject,
      html: emailContent.html,
      text: emailContent.text,
      tags,
    });

    if (!result.success) {
      return res.status(500).json({ 
        error: 'Failed to send email',
        details: result.details 
      });
    }

    return res.status(200).json({ 
      success: true,
      message: `Affiliate ${body.type} email sent`,
      id: result.id,
    });

  } catch (error) {
    console.error('Error sending affiliate email:', error);
    return res.status(500).json({ 
      error: 'Failed to send email',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

