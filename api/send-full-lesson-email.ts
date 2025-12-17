/**
 * Full Lesson Email API
 * 
 * Sends complete standalone lesson emails with both LEARN and GROW tracks.
 * Users can read the entire lesson via email without opening the app.
 * 
 * Environment Variables:
 * - RESEND_API_KEY: Your Resend API key
 * - DAILY_EMAIL_API_KEY: Secret key to authorize daily email sends
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

const RESEND_API_URL = 'https://api.resend.com/emails';

// Brand colors
const COLORS = {
  background: '#0a0a0b',
  cardBg: '#18181b',
  sectionBg: '#27272a',
  accent: '#3b82f6',
  learnAccent: '#f59e0b',  // Gold for LEARN
  growAccent: '#8b5cf6',   // Purple for GROW
  gold: '#fbbf24',
  text: '#f4f4f5',
  textMuted: '#a1a1aa',
  textDim: '#71717a',
  border: '#3f3f46',
};

interface LessonPhases {
  hook: string;
  fact1: string;
  fact2: string;
  fact3: string;
  wisdom: string;
}

interface FullLessonRequest {
  email: string;
  name?: string;
  dayNumber: number;
  
  // LEARN Track
  learnTitle: string;
  learnEmoji: string;
  learnCategory: string;
  learnPhases: LessonPhases;
  learnWisdom: string;
  learnFunFacts?: string[];
  
  // GROW Track
  growTitle: string;
  growEmoji?: string;
  growObjective: string;
  growContent: string;
  growActivity?: string;
  
  // Daily wisdom
  dailyWisdom?: string;
}

function generateFullLessonHTML(data: FullLessonRequest): string {
  const name = data.name || 'curious learner';
  const funFacts = data.learnFunFacts || [];
  
  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Day ${data.dayNumber} - Curious Kelly</title>
</head>
<body style="margin: 0; padding: 0; font-family: Georgia, 'Times New Roman', serif; background-color: ${COLORS.background}; color: ${COLORS.text};">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: ${COLORS.background}; padding: 40px 20px;">
    <tr>
      <td align="center">
        <table width="100%" style="max-width: 600px;">
          
          <!-- Header -->
          <tr>
            <td style="text-align: center; padding-bottom: 30px;">
              <p style="font-size: 14px; color: ${COLORS.textDim}; margin: 0 0 8px; font-family: -apple-system, sans-serif; text-transform: uppercase; letter-spacing: 2px;">
                ✨ CURIOUS KELLY — Day ${data.dayNumber}
              </p>
            </td>
          </tr>
          
          <!-- ═══════════════════════════════════════ -->
          <!-- LEARN TRACK -->
          <!-- ═══════════════════════════════════════ -->
          
          <tr>
            <td style="background-color: ${COLORS.cardBg}; border-radius: 16px; overflow: hidden; margin-bottom: 24px;">
              
              <!-- LEARN Header -->
              <table width="100%" style="background: linear-gradient(135deg, ${COLORS.learnAccent}22, ${COLORS.learnAccent}11);">
                <tr>
                  <td style="padding: 24px 30px; border-bottom: 1px solid ${COLORS.border};">
                    <p style="font-size: 12px; color: ${COLORS.learnAccent}; margin: 0 0 8px; font-family: -apple-system, sans-serif; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">
                      🌟 TODAY'S LEARN LESSON
                    </p>
                    <h1 style="font-size: 28px; font-weight: 700; margin: 0; color: ${COLORS.text};">
                      ${data.learnEmoji} ${data.learnTitle}
                    </h1>
                    <p style="font-size: 14px; color: ${COLORS.textMuted}; margin: 8px 0 0; font-family: -apple-system, sans-serif;">
                      ${data.learnCategory} • 5 minute read
                    </p>
                  </td>
                </tr>
              </table>
              
              <!-- LEARN Content -->
              <table width="100%">
                <tr>
                  <td style="padding: 30px;">
                    
                    <!-- Hook -->
                    <p style="font-size: 18px; line-height: 1.8; color: ${COLORS.text}; margin: 0 0 24px;">
                      ${data.learnPhases.hook}
                    </p>
                    
                    <!-- Fact 1 -->
                    <div style="background: ${COLORS.sectionBg}; border-radius: 12px; padding: 20px; margin-bottom: 20px; border-left: 4px solid ${COLORS.learnAccent};">
                      <p style="font-size: 12px; color: ${COLORS.learnAccent}; margin: 0 0 8px; font-family: -apple-system, sans-serif; font-weight: 600;">
                        💡 FACT ONE
                      </p>
                      <p style="font-size: 16px; line-height: 1.7; color: ${COLORS.text}; margin: 0;">
                        ${data.learnPhases.fact1}
                      </p>
                    </div>
                    
                    <!-- Fact 2 -->
                    <div style="background: ${COLORS.sectionBg}; border-radius: 12px; padding: 20px; margin-bottom: 20px; border-left: 4px solid ${COLORS.learnAccent};">
                      <p style="font-size: 12px; color: ${COLORS.learnAccent}; margin: 0 0 8px; font-family: -apple-system, sans-serif; font-weight: 600;">
                        🧠 FACT TWO
                      </p>
                      <p style="font-size: 16px; line-height: 1.7; color: ${COLORS.text}; margin: 0;">
                        ${data.learnPhases.fact2}
                      </p>
                    </div>
                    
                    <!-- Fact 3 -->
                    <div style="background: ${COLORS.sectionBg}; border-radius: 12px; padding: 20px; margin-bottom: 20px; border-left: 4px solid ${COLORS.learnAccent};">
                      <p style="font-size: 12px; color: ${COLORS.learnAccent}; margin: 0 0 8px; font-family: -apple-system, sans-serif; font-weight: 600;">
                        ✨ FACT THREE
                      </p>
                      <p style="font-size: 16px; line-height: 1.7; color: ${COLORS.text}; margin: 0;">
                        ${data.learnPhases.fact3}
                      </p>
                    </div>
                    
                    <!-- Wisdom -->
                    <div style="background: linear-gradient(135deg, ${COLORS.gold}15, ${COLORS.gold}05); border-radius: 12px; padding: 24px; margin-top: 24px; border: 1px solid ${COLORS.gold}33;">
                      <p style="font-size: 12px; color: ${COLORS.gold}; margin: 0 0 8px; font-family: -apple-system, sans-serif; font-weight: 600;">
                        🦉 TODAY'S WISDOM
                      </p>
                      <p style="font-size: 17px; line-height: 1.8; color: ${COLORS.text}; margin: 0; font-style: italic;">
                        ${data.learnPhases.wisdom}
                      </p>
                    </div>
                    
                    ${funFacts.length > 0 ? `
                    <!-- Fun Facts -->
                    <div style="margin-top: 24px; padding-top: 24px; border-top: 1px solid ${COLORS.border};">
                      <p style="font-size: 12px; color: ${COLORS.textMuted}; margin: 0 0 12px; font-family: -apple-system, sans-serif; font-weight: 600;">
                        💫 DID YOU KNOW?
                      </p>
                      ${funFacts.map(fact => `
                      <p style="font-size: 14px; line-height: 1.6; color: ${COLORS.textMuted}; margin: 0 0 8px; padding-left: 16px; border-left: 2px solid ${COLORS.border};">
                        ${fact}
                      </p>
                      `).join('')}
                    </div>
                    ` : ''}
                    
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          
          <!-- Spacer -->
          <tr><td style="height: 24px;"></td></tr>
          
          <!-- ═══════════════════════════════════════ -->
          <!-- GROW TRACK -->
          <!-- ═══════════════════════════════════════ -->
          
          <tr>
            <td style="background-color: ${COLORS.cardBg}; border-radius: 16px; overflow: hidden;">
              
              <!-- GROW Header -->
              <table width="100%" style="background: linear-gradient(135deg, ${COLORS.growAccent}22, ${COLORS.growAccent}11);">
                <tr>
                  <td style="padding: 24px 30px; border-bottom: 1px solid ${COLORS.border};">
                    <p style="font-size: 12px; color: ${COLORS.growAccent}; margin: 0 0 8px; font-family: -apple-system, sans-serif; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">
                      🧠 TODAY'S GROW LESSON
                    </p>
                    <h2 style="font-size: 24px; font-weight: 700; margin: 0; color: ${COLORS.text};">
                      ${data.growEmoji || '🌱'} ${data.growTitle}
                    </h2>
                    <p style="font-size: 14px; color: ${COLORS.textMuted}; margin: 8px 0 0; font-family: -apple-system, sans-serif;">
                      Meta-Learning • ${data.growObjective}
                    </p>
                  </td>
                </tr>
              </table>
              
              <!-- GROW Content -->
              <table width="100%">
                <tr>
                  <td style="padding: 30px;">
                    <p style="font-size: 17px; line-height: 1.8; color: ${COLORS.text}; margin: 0 0 24px;">
                      ${data.growContent}
                    </p>
                    
                    ${data.growActivity ? `
                    <!-- Activity -->
                    <div style="background: ${COLORS.sectionBg}; border-radius: 12px; padding: 20px; border-left: 4px solid ${COLORS.growAccent};">
                      <p style="font-size: 12px; color: ${COLORS.growAccent}; margin: 0 0 8px; font-family: -apple-system, sans-serif; font-weight: 600;">
                        🎯 TRY THIS TODAY
                      </p>
                      <p style="font-size: 16px; line-height: 1.7; color: ${COLORS.text}; margin: 0;">
                        ${data.growActivity}
                      </p>
                    </div>
                    ` : ''}
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          
          <!-- Spacer -->
          <tr><td style="height: 24px;"></td></tr>
          
          <!-- ═══════════════════════════════════════ -->
          <!-- DAILY WISDOM / CTA -->
          <!-- ═══════════════════════════════════════ -->
          
          ${data.dailyWisdom ? `
          <tr>
            <td style="text-align: center; padding: 30px;">
              <p style="font-size: 12px; color: ${COLORS.gold}; margin: 0 0 12px; font-family: -apple-system, sans-serif; text-transform: uppercase; letter-spacing: 2px;">
                ✨ DAILY WISDOM
              </p>
              <p style="font-size: 20px; line-height: 1.6; color: ${COLORS.text}; margin: 0; font-style: italic;">
                "${data.dailyWisdom}"
              </p>
            </td>
          </tr>
          ` : ''}
          
          <!-- CTA -->
          <tr>
            <td style="text-align: center; padding: 20px;">
              <a href="https://curiouskelly.com/day/${data.dayNumber}" style="display: inline-block; background: ${COLORS.accent}; color: white; text-decoration: none; padding: 16px 32px; border-radius: 12px; font-family: -apple-system, sans-serif; font-weight: 600; font-size: 16px;">
                Experience with Kelly →
              </a>
              <p style="font-size: 13px; color: ${COLORS.textDim}; margin: 12px 0 0;">
                Watch today's lesson with voice & video
              </p>
            </td>
          </tr>
          
          <!-- Footer -->
          <tr>
            <td style="text-align: center; padding: 40px 20px 20px; border-top: 1px solid ${COLORS.border}; margin-top: 30px;">
              <p style="font-size: 14px; color: ${COLORS.textMuted}; margin: 0 0 8px;">
                Stay curious! ✨<br>
                — Kelly
              </p>
              <p style="font-size: 12px; color: ${COLORS.textDim}; margin: 20px 0 0;">
                <a href="https://curiouskelly.com" style="color: ${COLORS.textDim};">curiouskelly.com</a> • 
                <a href="https://curiouskelly.com/help" style="color: ${COLORS.textDim};">Help</a> • 
                <a href="https://curiouskelly.com/api/unsubscribe" style="color: ${COLORS.textDim};">Unsubscribe</a>
              </p>
              <p style="font-size: 11px; color: ${COLORS.textDim}; margin: 12px 0 0;">
                © 2025 Lesson of the Day PBC
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

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const authHeader = req.headers.authorization;
  const expectedKey = process.env.DAILY_EMAIL_API_KEY;
  
  if (expectedKey && authHeader !== `Bearer ${expectedKey}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const resendApiKey = process.env.RESEND_API_KEY;
  if (!resendApiKey) {
    return res.status(500).json({ error: 'Email service not configured' });
  }

  try {
    const body = req.body as FullLessonRequest;
    
    // Validate required fields
    if (!body.email || !body.dayNumber || !body.learnTitle || !body.learnPhases) {
      return res.status(400).json({ 
        error: 'Missing required fields',
        required: ['email', 'dayNumber', 'learnTitle', 'learnPhases', 'growTitle', 'growContent']
      });
    }

    const html = generateFullLessonHTML(body);
    const subject = `Day ${body.dayNumber}: ${body.learnEmoji} ${body.learnTitle} + 🧠 ${body.growTitle}`;

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
        tags: [{ name: 'type', value: 'full_lesson' }],
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      return res.status(500).json({ error: 'Failed to send email', details: data });
    }

    return res.status(200).json({ 
      success: true,
      message: 'Full lesson email sent',
      id: data.id,
    });

  } catch (error) {
    console.error('Error sending full lesson email:', error);
    return res.status(500).json({ 
      error: 'Failed to send email',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}
