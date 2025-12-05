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
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

Hi — I'm Kelly.<br><br>

I don't have all the answers. But I love finding them. And I think learning is better together.<br><br>

Every day I find something wonderful and I can't wait to share it. Today's lesson is ready.<br><br>

<a href="${lessonUrl}" style="color: #1e3a5f; text-decoration: underline;">Want to come along?</a><br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
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


