/**
 * Send Daily Duo Email — V2 Template
 * 
 * Kelly DELIVERS the lesson IN the email.
 * Full content, real images, no need to click.
 * 
 * POST /api/email/send-daily-duo
 * Body: { email: string, day?: number, name?: string }
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { Resend } from 'resend';

const resend = new Resend(process.env.RESEND_API_KEY);

const BASE_URL = 'https://www.curiouskelly.com';

// Kelly images for email
const KELLY_IMAGES = {
  hero: `${BASE_URL}/images/kelly-og-image.png`,
  avatar: `${BASE_URL}/images/kelly/kelly-chair-curious.png`,
  explaining: `${BASE_URL}/images/kelly/kelly-chair-explaining.png`,
  wisdom: `${BASE_URL}/images/kelly/kelly-chair-wisdom.png`,
  celebrating: `${BASE_URL}/images/kelly/kelly-chair-celebrating.png`,
};

interface DailyDuoRequest {
  email: string;
  day?: number;
  name?: string;
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
): Promise<void> {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  try {
    const { email, day, name } = req.body as DailyDuoRequest;

    if (!email) {
      res.status(400).json({ error: 'Email required' });
      return;
    }

    const dayNumber = day || getTodayDayNumber();
    const learnerName = name || 'Curious Learner';

    // Fetch content
    const contentResponse = await fetch(`${BASE_URL}/api/email/daily-content?day=${dayNumber}`);
    
    if (!contentResponse.ok) {
      throw new Error('Failed to fetch lesson content');
    }

    const content = await contentResponse.json();

    // Generate V2 emails
    const htmlEmail = generateDailyDuoHtmlV2(content, learnerName);
    const textEmail = generateDailyDuoText(content, learnerName);

    // Send via Resend
    const { data, error } = await resend.emails.send({
      from: 'Kelly <hello@curiouskelly.com>',
      to: email,
      subject: `✨ Day ${dayNumber}: ${content.learn.topic} + ${content.grow.topic}`,
      html: htmlEmail,
      text: textEmail
    });

    if (error) {
      console.error('[send-daily-duo] Resend error:', error);
      res.status(500).json({ error: 'Failed to send email' });
      return;
    }

    res.status(200).json({ 
      success: true, 
      messageId: data?.id,
      day: dayNumber,
      version: 'v2',
      topics: {
        learn: content.learn.topic,
        grow: content.grow.topic
      }
    });
  } catch (error) {
    console.error('[send-daily-duo] Error:', error);
    res.status(500).json({ error: 'Failed to send daily duo email' });
  }
}

function getTodayDayNumber(): number {
  const now = new Date();
  const start = new Date(now.getFullYear(), 0, 0);
  const diff = now.getTime() - start.getTime();
  const oneDay = 1000 * 60 * 60 * 24;
  return Math.floor(diff / oneDay);
}

/**
 * V2 Email Template — Kelly Delivers IN the Email
 */
function generateDailyDuoHtmlV2(content: any, name: string): string {
  const { day, date, learn, grow, combined_wisdom } = content;

  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Day ${day}: ${learn.topic}</title>
  <!--[if mso]>
  <noscript>
    <xml>
      <o:OfficeDocumentSettings>
        <o:PixelsPerInch>96</o:PixelsPerInch>
      </o:OfficeDocumentSettings>
    </xml>
  </noscript>
  <![endif]-->
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f5f5f5;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color: #f5f5f5;">
    <tr>
      <td align="center" style="padding: 20px;">
        <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 0; overflow: hidden;">
          
          <!-- Hero with Kelly -->
          <tr>
            <td style="background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%); text-align: center; padding: 0;">
              <img src="${KELLY_IMAGES.hero}" alt="Kelly" width="600" style="width: 100%; max-width: 600px; height: auto; display: block;" />
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td style="padding: 20px 24px 24px; text-align: center;">
                    <span style="display: inline-block; background: linear-gradient(135deg, #f59e0b, #8b5cf6); color: white; padding: 6px 16px; border-radius: 20px; font-size: 12px; font-weight: 700;">Day ${day} — ${date}</span>
                    <h1 style="color: white; font-size: 22px; font-weight: 600; margin: 12px 0 8px;">Good morning, ${name}!</h1>
                    <p style="color: rgba(255,255,255,0.8); font-size: 14px; margin: 0;">Your Daily Duo is ready ✨</p>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- Kelly Says Hello -->
          <tr>
            <td style="background: #f8fafc; padding: 24px; border-bottom: 1px solid #e2e8f0;">
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td width="70" valign="top">
                    <img src="${KELLY_IMAGES.avatar}" alt="Kelly" width="56" height="56" style="border-radius: 50%; border: 3px solid #3b82f6;" />
                  </td>
                  <td valign="top" style="padding-left: 14px;">
                    <div style="background: white; border-radius: 16px; border-top-left-radius: 4px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
                      <div style="font-size: 12px; font-weight: 700; color: #3b82f6; margin-bottom: 6px;">Kelly ✨</div>
                      <div style="font-size: 15px; color: #334155; line-height: 1.5;">
                        Today we're exploring something amazing — ${learn.topic.toLowerCase()}. Plus, I'll share a skill that will help you learn anything faster. Let's dive in!
                      </div>
                    </div>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- LEARN Section -->
          <tr>
            <td style="padding: 28px 24px;">
              <span style="display: inline-block; background: #fef3c7; color: #92400e; padding: 6px 12px; border-radius: 6px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">🌟 LEARN</span>
              <h2 style="font-size: 24px; font-weight: 700; color: #0f172a; margin: 0 0 16px; line-height: 1.2;">${learn.topic}</h2>
              
              <!-- Lesson Image -->
              <img src="${KELLY_IMAGES.explaining}" alt="Kelly explaining" width="552" style="width: 100%; max-width: 552px; height: auto; border-radius: 12px; margin-bottom: 20px;" />
              
              <!-- Lesson Content -->
              <div style="font-size: 16px; color: #334155; line-height: 1.7;">
                ${learn.hook ? `<p style="margin: 0 0 16px;">${learn.hook}</p>` : ''}
                
                <!-- Kelly Explains -->
                <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin: 20px 0;">
                  <tr>
                    <td style="background: #f0f9ff; border-left: 4px solid #3b82f6; padding: 16px; border-radius: 0 8px 8px 0;">
                      <table role="presentation" cellpadding="0" cellspacing="0">
                        <tr>
                          <td width="36" valign="top">
                            <img src="${KELLY_IMAGES.avatar}" alt="Kelly" width="28" height="28" style="border-radius: 50%; border: 2px solid #3b82f6;" />
                          </td>
                          <td valign="top" style="padding-left: 8px;">
                            <div style="font-size: 11px; font-weight: 700; color: #3b82f6; text-transform: uppercase; margin-bottom: 4px;">Kelly explains</div>
                            <div style="font-size: 15px; color: #1e40af; font-style: italic;">
                              "${learn.facts[0] || 'This is one of the most fascinating things I\'ve learned!'}"
                            </div>
                          </td>
                        </tr>
                      </table>
                    </td>
                  </tr>
                </table>
                
                ${learn.facts.slice(1).map((fact: string) => `<p style="margin: 0 0 16px;">${fact}</p>`).join('')}
                
                <!-- Universal Truth -->
                ${learn.universal_truth ? `
                <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin: 20px 0;">
                  <tr>
                    <td style="background: linear-gradient(135deg, #fef3c7, #fde68a); padding: 20px; border-radius: 12px; text-align: center;">
                      <div style="font-size: 28px; margin-bottom: 8px;">💡</div>
                      <div style="font-size: 18px; font-weight: 600; color: #78350f; line-height: 1.4;">${learn.universal_truth}</div>
                    </td>
                  </tr>
                </table>
                ` : ''}
                
                <!-- Question -->
                ${learn.question ? `
                <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin: 20px 0;">
                  <tr>
                    <td style="background: #fffbeb; border: 2px solid #fcd34d; border-radius: 12px; padding: 20px;">
                      <div style="font-size: 12px; font-weight: 700; color: #92400e; text-transform: uppercase; margin-bottom: 8px;">🤔 Think about it</div>
                      <div style="font-size: 16px; color: #78350f; font-weight: 500;">${learn.question}</div>
                    </td>
                  </tr>
                </table>
                ` : ''}
              </div>
            </td>
          </tr>

          <!-- Divider -->
          <tr>
            <td style="padding: 0 24px;">
              <div style="height: 1px; background: linear-gradient(to right, transparent, #e2e8f0, transparent);"></div>
            </td>
          </tr>

          <!-- GROW Section -->
          <tr>
            <td style="padding: 28px 24px; background: #faf5ff;">
              <span style="display: inline-block; background: #ede9fe; color: #5b21b6; padding: 6px 12px; border-radius: 6px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">🧠 GROW</span>
              <h2 style="font-size: 24px; font-weight: 700; color: #0f172a; margin: 0 0 16px; line-height: 1.2;">${grow.topic}</h2>
              
              <!-- Skill Image -->
              <img src="${KELLY_IMAGES.wisdom}" alt="Kelly teaching" width="552" style="width: 100%; max-width: 552px; height: auto; border-radius: 12px; margin-bottom: 20px;" />
              
              <div style="font-size: 16px; color: #334155; line-height: 1.7;">
                <p style="margin: 0 0 16px;"><strong>Today's skill:</strong> ${grow.objective}</p>
                
                <!-- Kelly's Tip -->
                <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin: 20px 0;">
                  <tr>
                    <td style="background: #faf5ff; border-left: 4px solid #8b5cf6; padding: 16px; border-radius: 0 8px 8px 0;">
                      <table role="presentation" cellpadding="0" cellspacing="0">
                        <tr>
                          <td width="36" valign="top">
                            <img src="${KELLY_IMAGES.avatar}" alt="Kelly" width="28" height="28" style="border-radius: 50%; border: 2px solid #8b5cf6;" />
                          </td>
                          <td valign="top" style="padding-left: 8px;">
                            <div style="font-size: 11px; font-weight: 700; color: #7c3aed; text-transform: uppercase; margin-bottom: 4px;">Kelly's tip</div>
                            <div style="font-size: 15px; color: #5b21b6; font-style: italic;">
                              "The best learners don't just practice — they reflect. After each attempt, ask: What worked? What didn't? What will I try differently?"
                            </div>
                          </td>
                        </tr>
                      </table>
                    </td>
                  </tr>
                </table>
                
                ${grow.content ? `<p style="margin: 0 0 16px;">${grow.content}</p>` : ''}
                
                <!-- Try This -->
                <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin: 20px 0;">
                  <tr>
                    <td style="background: #f3e8ff; border: 2px solid #c4b5fd; border-radius: 12px; padding: 20px;">
                      <div style="font-size: 12px; font-weight: 700; color: #6b21a8; text-transform: uppercase; margin-bottom: 8px;">⚡ Try this today</div>
                      <div style="font-size: 16px; color: #581c87; font-weight: 500;">After learning something new, take 60 seconds to write down one thing you understood well and one thing that's still fuzzy.</div>
                    </td>
                  </tr>
                </table>
              </div>
            </td>
          </tr>

          <!-- Daily Wisdom -->
          <tr>
            <td style="background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%); padding: 32px 24px; text-align: center;">
              <img src="${KELLY_IMAGES.celebrating}" alt="Kelly" width="64" height="64" style="border-radius: 50%; border: 3px solid #fbbf24; margin-bottom: 16px;" />
              <div style="font-size: 11px; font-weight: 700; color: #fbbf24; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px;">✨ Today's Wisdom</div>
              <div style="font-size: 20px; font-weight: 500; color: white; line-height: 1.5; font-style: italic;">
                "${combined_wisdom}"
              </div>
              <div style="margin-top: 16px; font-size: 14px; color: rgba(255,255,255,0.7);">— Kelly 💙</div>
            </td>
          </tr>

          <!-- Optional CTA (soft) -->
          <tr>
            <td style="padding: 24px; background: #f8fafc; text-align: center; border-top: 1px solid #e2e8f0;">
              <div style="font-size: 13px; color: #64748b; margin-bottom: 12px;">Want to hear me explain this with voice and video?</div>
              <a href="${BASE_URL}/learn.html?day=${day}" style="display: inline-block; background: #e2e8f0; color: #475569; text-decoration: none; padding: 10px 20px; border-radius: 6px; font-size: 13px; font-weight: 500;">
                Watch Today's Lesson →
              </a>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="padding: 24px; text-align: center; font-size: 12px; color: #94a3b8;">
              <div style="font-size: 16px; margin-bottom: 8px;">✨ Curious Kelly</div>
              <p style="margin: 0 0 12px;">You received your complete lesson. No need to click anything.</p>
              <p style="margin: 0;">
                <a href="${BASE_URL}/preferences" style="color: #64748b; text-decoration: none;">Preferences</a> · 
                <a href="${BASE_URL}/unsubscribe" style="color: #64748b; text-decoration: none;">Unsubscribe</a> · 
                <a href="${BASE_URL}" style="color: #64748b; text-decoration: none;">curiouskelly.com</a>
              </p>
              <p style="margin: 12px 0 0; color: #cbd5e1;">
                Lesson of the Day PBC · hello@curiouskelly.com
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>
`;
}

function generateDailyDuoText(content: any, name: string): string {
  const { day, date, learn, grow, combined_wisdom } = content;

  return `
✨ CURIOUS KELLY — Day ${day}
${date}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Good morning, ${name}!

Here's your Daily Duo — two lessons to make today count.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌟 LEARN: ${learn.topic}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

${learn.hook || ''}

${learn.facts.join('\n\n')}

💡 ${learn.universal_truth || ''}

${learn.question ? `🤔 Think about it: ${learn.question}` : ''}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 GROW: ${grow.topic}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Today's skill: ${grow.objective}

${grow.content || ''}

⚡ Try this: After learning something new, take 60 seconds to write down one thing you understood well and one thing that's still fuzzy.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ TODAY'S WISDOM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"${combined_wisdom}"

— Kelly 💙

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You received your complete lesson. No need to click anything.

But if you want voice & video:
${BASE_URL}/learn.html?day=${day}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stay curious! ✨
— Kelly

Preferences: ${BASE_URL}/preferences
Unsubscribe: ${BASE_URL}/unsubscribe

Lesson of the Day PBC · hello@curiouskelly.com
`.trim();
}
