/**
 * Welcome Email API Endpoint
 * 
 * Sends a personalized welcome email when a user signs up.
 * Triggered by Supabase auth webhook or called directly after signup.
 * 
 * Environment Variables Required:
 * - RESEND_API_KEY: Your Resend API key
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

// Resend API endpoint
const RESEND_API_URL = 'https://api.resend.com/emails';

// Fun facts to rotate in emails
const FUN_FACTS = [
  "The average person stops actively learning around age 25. You just changed that!",
  "You create 700 new neural connections every time you learn something new.",
  "Children ask ~300 questions per day. Adults? About 20. Let's change that!",
  "Reading 20 minutes a day exposes you to 1.8 million words per year.",
  "Einstein failed his university entrance exam the first time. Look where curiosity took him!",
  "Your brain uses 20% of your body's energy but is only 2% of your weight.",
  "The word 'curious' comes from Latin 'cura' meaning 'care' - to be curious is to care!",
  "It takes an average of 66 days to form a habit. Day 1 starts now!",
];

function getRandomFact(): string {
  return FUN_FACTS[Math.floor(Math.random() * FUN_FACTS.length)];
}

function generateWelcomeEmailHTML(name: string, lessonUrl: string): string {
  // Kelly's Voice: Humble, Curious, Collaborative, Warm, Simple, Rich
  return `
<div style="max-width: 480px; font-family: Georgia, serif;">
  <p style="text-align: center; margin: 0 0 24px 0;">
    <img src="https://curiouskelly.com/images/brand/kelly-mark-circle-128.png" alt="Kelly" width="80" height="80" style="border-radius: 50%; border: 3px solid #3b82f6;">
  </p>
  
  <p style="font-size: 19px; color: #1f2937; line-height: 1.9; margin: 0 0 20px;">
    Hi — I'm Kelly.
  </p>
  
  <p style="font-size: 17px; color: #374151; line-height: 1.9; margin: 0 0 20px;">
    I don't have all the answers. But I love finding them. And I think learning is better together.
  </p>
  
  <p style="font-size: 17px; color: #374151; line-height: 1.9; margin: 0 0 20px;">
    Every day I find something wonderful and I can't wait to share it. Today's lesson is ready.
  </p>
  
  <p style="margin: 0 0 24px 0;">
    <a href="${lessonUrl}" style="display: inline-block; background: #3b82f6; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-family: -apple-system, sans-serif; font-size: 15px; font-weight: 500;">
      Want to come along? →
    </a>
  </p>
  
  <p style="font-size: 15px; color: #6b7280; font-style: italic; margin: 0;">
    — Kelly
  </p>
  
  <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 24px 0;">
  
  <p style="font-family: -apple-system, sans-serif; font-size: 11px; color: #9ca3af; margin: 0; text-align: center;">
    ✨ Curious Kelly · <a href="https://curiouskelly.com" style="color: #9ca3af;">curiouskelly.com</a><br>
    Lesson of the Day PBC · hello@curiouskelly.com
  </p>
</div>
  `.trim();
}

function generateWelcomeEmailText(name: string, lessonUrl: string): string {
  // Kelly's Voice: Humble, Curious, Collaborative, Warm, Simple, Rich
  return `
Hi — I'm Kelly.

I don't have all the answers. But I love finding them. And I think learning is better together.

Every day I find something wonderful and I can't wait to share it. Today's lesson is ready.

Want to come along? ${lessonUrl}

— Kelly
  `.trim();
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Verify API key (simple auth)
  const authHeader = req.headers.authorization;
  const expectedKey = process.env.WELCOME_EMAIL_API_KEY;
  
  if (expectedKey && authHeader !== `Bearer ${expectedKey}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  // Get request body
  const { email, name } = req.body;

  if (!email) {
    return res.status(400).json({ error: 'Email is required' });
  }

  // Get Resend API key
  const resendApiKey = process.env.RESEND_API_KEY;
  
  if (!resendApiKey) {
    console.error('RESEND_API_KEY not configured');
    return res.status(500).json({ error: 'Email service not configured' });
  }

  // Default name if not provided
  const displayName = name || 'Curious Friend';
  const lessonUrl = 'https://curiouskelly.com/learn';

  try {
    const response = await fetch(RESEND_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${resendApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'Kelly <hello@curiouskelly.com>',
        to: email,
        subject: `Welcome to curiosity, ${displayName}! 🎉`,
        html: generateWelcomeEmailHTML(displayName, lessonUrl),
        text: generateWelcomeEmailText(displayName, lessonUrl),
        reply_to: 'hello@curiouskelly.com',
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      console.error('Resend API error:', data);
      return res.status(response.status).json({ 
        error: 'Failed to send email',
        details: data 
      });
    }

    console.log(`Welcome email sent to ${email}`, data);
    
    return res.status(200).json({ 
      success: true,
      message: 'Welcome email sent',
      id: data.id 
    });

  } catch (error) {
    console.error('Error sending welcome email:', error);
    return res.status(500).json({ 
      error: 'Failed to send email',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}


