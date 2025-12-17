/**
 * Send Daily Duo Email
 * 
 * Sends the full LEARN + GROW email to a subscriber
 * Uses the content extraction API to get lesson content
 * 
 * POST /api/email/send-daily-duo
 * Body: { email: string, day?: number, name?: string }
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { Resend } from 'resend';

const resend = new Resend(process.env.RESEND_API_KEY);

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

    // Get today's day number if not provided
    const dayNumber = day || getTodayDayNumber();
    const learnerName = name || 'Curious Learner';

    // Fetch content from our extraction API
    const baseUrl = process.env.VERCEL_URL 
      ? `https://${process.env.VERCEL_URL}` 
      : 'https://www.curiouskelly.com';
    
    const contentResponse = await fetch(`${baseUrl}/api/email/daily-content?day=${dayNumber}`);
    
    if (!contentResponse.ok) {
      throw new Error('Failed to fetch lesson content');
    }

    const content = await contentResponse.json();

    // Generate HTML email
    const htmlEmail = generateDailyDuoHtml(content, learnerName);
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

function generateDailyDuoHtml(content: any, name: string): string {
  const { day, date, learn, grow, combined_wisdom } = content;

  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Day ${day}: ${learn.topic}</title>
  <style>
    body { 
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      line-height: 1.6;
      color: #1a1a1a;
      max-width: 600px;
      margin: 0 auto;
      padding: 20px;
      background: #f5f5f5;
    }
    .container {
      background: white;
      border-radius: 16px;
      padding: 32px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .header {
      text-align: center;
      border-bottom: 2px solid #f0f0f0;
      padding-bottom: 20px;
      margin-bottom: 24px;
    }
    .logo { font-size: 28px; margin-bottom: 8px; }
    .day-badge {
      display: inline-block;
      background: linear-gradient(135deg, #f59e0b, #8b5cf6);
      color: white;
      padding: 8px 16px;
      border-radius: 20px;
      font-weight: 600;
      font-size: 14px;
    }
    .section {
      margin: 28px 0;
      padding: 20px;
      border-radius: 12px;
    }
    .learn-section {
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.08), rgba(245, 158, 11, 0.04));
      border-left: 4px solid #f59e0b;
    }
    .grow-section {
      background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(139, 92, 246, 0.04));
      border-left: 4px solid #8b5cf6;
    }
    .section-header {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 12px;
    }
    .section-icon { font-size: 24px; }
    .section-label {
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    .learn-label { color: #f59e0b; }
    .grow-label { color: #8b5cf6; }
    .section-title {
      font-size: 20px;
      font-weight: 600;
      margin: 0 0 12px 0;
      color: #1a1a1a;
    }
    .section-content {
      font-size: 16px;
      color: #444;
    }
    .universal-truth {
      font-style: italic;
      color: #666;
      margin-top: 16px;
      padding: 12px;
      background: rgba(255,255,255,0.6);
      border-radius: 8px;
    }
    .question-box {
      background: #fffbeb;
      border: 1px solid #fcd34d;
      border-radius: 8px;
      padding: 16px;
      margin-top: 16px;
    }
    .question-label {
      font-size: 12px;
      font-weight: 600;
      color: #92400e;
      margin-bottom: 4px;
    }
    .wisdom-section {
      text-align: center;
      padding: 24px;
      background: linear-gradient(135deg, #fef3c7, #e0e7ff);
      border-radius: 12px;
      margin: 28px 0;
    }
    .wisdom-icon { font-size: 32px; margin-bottom: 12px; }
    .wisdom-text {
      font-size: 18px;
      font-weight: 500;
      color: #1a1a1a;
      font-style: italic;
    }
    .cta-section {
      text-align: center;
      margin: 28px 0;
    }
    .cta-button {
      display: inline-block;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      color: white;
      text-decoration: none;
      padding: 14px 28px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 16px;
    }
    .cta-subtitle {
      font-size: 13px;
      color: #666;
      margin-top: 8px;
    }
    .footer {
      text-align: center;
      padding-top: 20px;
      border-top: 1px solid #eee;
      margin-top: 28px;
      font-size: 13px;
      color: #888;
    }
    .footer a { color: #666; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="logo">✨ Curious Kelly</div>
      <div class="day-badge">Day ${day} — ${date}</div>
    </div>

    <p>Good morning, ${name}! ☀️</p>
    <p>Here's your Daily Duo — two lessons to make today count.</p>

    <!-- LEARN Section -->
    <div class="section learn-section">
      <div class="section-header">
        <span class="section-icon">🌟</span>
        <span class="section-label learn-label">LEARN</span>
      </div>
      <h2 class="section-title">${learn.topic}</h2>
      <div class="section-content">
        ${learn.hook ? `<p>${learn.hook}</p>` : ''}
        ${learn.facts.map((fact: string) => `<p>${fact}</p>`).join('')}
        ${learn.universal_truth ? `
          <div class="universal-truth">
            💡 ${learn.universal_truth}
          </div>
        ` : ''}
        ${learn.question ? `
          <div class="question-box">
            <div class="question-label">🤔 Think about it:</div>
            <div>${learn.question}</div>
          </div>
        ` : ''}
      </div>
    </div>

    <!-- GROW Section -->
    <div class="section grow-section">
      <div class="section-header">
        <span class="section-icon">🧠</span>
        <span class="section-label grow-label">GROW</span>
      </div>
      <h2 class="section-title">${grow.topic}</h2>
      <div class="section-content">
        <p><strong>Today's skill:</strong> ${grow.objective}</p>
        ${grow.content ? `<p>${grow.content}</p>` : ''}
      </div>
    </div>

    <!-- Wisdom -->
    <div class="wisdom-section">
      <div class="wisdom-icon">✨</div>
      <div class="wisdom-text">"${combined_wisdom}"</div>
    </div>

    <!-- CTA -->
    <div class="cta-section">
      <a href="https://www.curiouskelly.com/learn.html?day=${day}" class="cta-button">
        Experience with Kelly →
      </a>
      <div class="cta-subtitle">Watch today's lesson with voice &amp; video</div>
    </div>

    <div class="footer">
      <p>Stay curious! ✨<br>— Kelly</p>
      <p>
        <a href="https://www.curiouskelly.com/unsubscribe">Unsubscribe</a> · 
        <a href="https://www.curiouskelly.com/preferences">Preferences</a> · 
        <a href="https://www.curiouskelly.com">curiouskelly.com</a>
      </p>
      <p style="color: #aaa; font-size: 11px;">
        Lesson of the Day PBC · hello@curiouskelly.com
      </p>
    </div>
  </div>
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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ DAILY WISDOM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"${combined_wisdom}"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Experience with Kelly (voice & video):
https://www.curiouskelly.com/learn.html?day=${day}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stay curious! ✨
— Kelly

Unsubscribe: https://www.curiouskelly.com/unsubscribe
Preferences: https://www.curiouskelly.com/preferences

Lesson of the Day PBC · hello@curiouskelly.com
`.trim();
}
