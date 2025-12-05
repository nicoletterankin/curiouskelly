/**
 * Curious Kelly Email Templates
 * 
 * Central repository for all email templates.
 * Every email should feel like it's personally from Kelly.
 * 
 * @module email-templates
 */

// ============================================================================
// SHARED STYLES & COMPONENTS
// ============================================================================

const BRAND_COLORS = {
  background: '#0a0a0b',
  cardBg: '#18181b',
  accent: '#3b82f6',
  gold: '#fbbf24',
  text: '#f4f4f5',
  textMuted: '#a1a1aa',
  textDim: '#71717a',
  border: '#27272a',
};

const emailWrapper = (content: string) => `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: ${BRAND_COLORS.background}; color: ${BRAND_COLORS.text};">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: ${BRAND_COLORS.background}; padding: 40px 20px;">
    <tr>
      <td align="center">
        <table width="100%" style="max-width: 560px; background-color: ${BRAND_COLORS.cardBg}; border-radius: 16px; overflow: hidden;">
          ${content}
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
`.trim();

const emailHeader = (title?: string) => `
<tr>
  <td style="padding: 40px 40px 20px; text-align: center;">
    <img src="https://curiouskelly.com/images/brand/kelly-mark-circle-64.png" alt="Kelly" width="80" height="80" style="border-radius: 50%; border: 3px solid ${BRAND_COLORS.accent};">
    ${title ? `<h1 style="color: ${BRAND_COLORS.text}; font-size: 28px; font-weight: 600; margin: 20px 0 0; text-align: center;">${title}</h1>` : ''}
  </td>
</tr>
`;

const emailFooter = () => `
<tr>
  <td style="padding: 30px 40px; border-top: 1px solid ${BRAND_COLORS.border}; text-align: center;">
    <p style="color: #52525b; font-size: 12px; margin: 0 0 10px;">
      ✨ Curious Kelly | Learn something new every day
    </p>
    <p style="color: #52525b; font-size: 12px; margin: 0;">
      <a href="https://curiouskelly.com" style="color: #52525b;">curiouskelly.com</a> · 
      <a href="https://curiouskelly.com/help" style="color: #52525b;">Help</a> · 
      <a href="https://curiouskelly.com/privacy" style="color: #52525b;">Privacy</a>
    </p>
    <p style="color: #3f3f46; font-size: 11px; margin: 15px 0 0;">
      © 2025 Lesson of the Day PBC
    </p>
  </td>
</tr>
`;

const ctaButton = (text: string, url: string, color = BRAND_COLORS.accent) => `
<table width="100%" cellpadding="0" cellspacing="0">
  <tr>
    <td align="center" style="padding: 20px 0;">
      <a href="${url}" style="display: inline-block; background-color: ${color}; color: white; text-decoration: none; padding: 14px 32px; border-radius: 8px; font-weight: 600; font-size: 16px;">
        ${text}
      </a>
    </td>
  </tr>
</table>
`;

const funFactBox = (fact: string) => `
<tr>
  <td style="padding: 0 40px 40px;">
    <table width="100%" style="background-color: ${BRAND_COLORS.border}; border-radius: 12px;">
      <tr>
        <td style="padding: 20px;">
          <p style="color: ${BRAND_COLORS.gold}; font-size: 14px; font-weight: 600; margin: 0 0 8px;">
            💡 Did you know?
          </p>
          <p style="color: ${BRAND_COLORS.textMuted}; font-size: 14px; line-height: 1.5; margin: 0;">
            ${fact}
          </p>
        </td>
      </tr>
    </table>
  </td>
</tr>
`;

// ============================================================================
// FUN FACTS COLLECTION
// ============================================================================

export const FUN_FACTS = [
  "The average person stops actively learning around age 25. You just changed that!",
  "You create 700 new neural connections every time you learn something new.",
  "Children ask ~300 questions per day. Adults? About 20. Let's change that!",
  "Reading 20 minutes a day exposes you to 1.8 million words per year.",
  "Einstein failed his university entrance exam the first time. Look where curiosity took him!",
  "Your brain uses 20% of your body's energy but is only 2% of your weight.",
  "The word 'curious' comes from Latin 'cura' meaning 'care' - to be curious is to care!",
  "It takes an average of 66 days to form a habit. Day 1 starts now!",
  "The human brain can process images in as little as 13 milliseconds.",
  "Learning a new skill can increase gray matter in your brain within days.",
  "Curiosity activates the brain's reward system, making learning feel good!",
  "The more you learn, the easier learning becomes - it's called neuroplasticity.",
];

export const getRandomFact = () => FUN_FACTS[Math.floor(Math.random() * FUN_FACTS.length)];

// ============================================================================
// DAILY LESSON TOPICS (Sample - would come from your lesson database)
// ============================================================================

export const SAMPLE_LESSONS = [
  { title: "How Money Works", emoji: "💰", category: "Economics" },
  { title: "Why the Sky is Blue", emoji: "🌤️", category: "Science" },
  { title: "The Secret Life of Trees", emoji: "🌳", category: "Nature" },
  { title: "How Your Phone Knows Where You Are", emoji: "📱", category: "Technology" },
  { title: "Why We Dream", emoji: "💭", category: "Psychology" },
];

// ============================================================================
// EMAIL TEMPLATES
// ============================================================================

// ----------------------------------------------------------------------------
// 1. WELCOME EMAIL (New User Signup)
// ----------------------------------------------------------------------------
export function welcomeEmail(name: string): { subject: string; html: string; text: string } {
  const displayName = name || 'Curious Friend';
  const fact = getRandomFact();
  
  const html = emailWrapper(`
    ${emailHeader(`Welcome to curiosity, ${displayName}! 🎉`)}
    <tr>
      <td style="padding: 0 40px 30px;">
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">
          Hey ${displayName}!
        </p>
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">
          I'm Kelly, and I'm SO excited you're here.
        </p>
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">
          You just joined thousands of curious minds who've decided that every day is a chance to learn something new. That's pretty amazing.
        </p>
        <p style="color: ${BRAND_COLORS.text}; font-size: 16px; line-height: 1.6; margin: 0 0 10px; font-weight: 600;">
          Here's what happens next:
        </p>
        <ul style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.8; margin: 0 0 20px; padding-left: 20px;">
          <li>Every day, I'll have a fresh lesson waiting for you</li>
          <li>Takes about 5 minutes (perfect with your morning coffee ☕)</li>
          <li>You'll finish each one knowing something you didn't before</li>
        </ul>
        ${ctaButton('Start Your First Lesson →', 'https://curiouskelly.com/learn')}
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 20px 0 0;">
          Can't wait to learn together!
        </p>
        <p style="color: ${BRAND_COLORS.text}; font-size: 16px; line-height: 1.6; margin: 20px 0 0;">
          ✨ Stay curious,<br><strong>Kelly</strong>
        </p>
        <p style="color: ${BRAND_COLORS.textDim}; font-size: 14px; margin: 20px 0 0;">
          P.S. Hit reply anytime - I read every message!
        </p>
      </td>
    </tr>
    ${funFactBox(fact)}
    ${emailFooter()}
  `);

  const text = `
Welcome to curiosity, ${displayName}! 🎉

Hey ${displayName}!

I'm Kelly, and I'm SO excited you're here.

You just joined thousands of curious minds who've decided that every day is a chance to learn something new. That's pretty amazing.

Here's what happens next:
→ Every day, I'll have a fresh lesson waiting for you
→ Takes about 5 minutes (perfect with your morning coffee ☕)
→ You'll finish each one knowing something you didn't before

Start your first lesson: https://curiouskelly.com/learn

Can't wait to learn together!

✨ Stay curious,
Kelly

P.S. Hit reply anytime - I read every message!

---
💡 Did you know? ${fact}

---
✨ Curious Kelly | Learn something new every day
curiouskelly.com | © 2025 Lesson of the Day PBC
  `.trim();

  return {
    subject: `Welcome to curiosity, ${displayName}! 🎉`,
    html,
    text,
  };
}

// ----------------------------------------------------------------------------
// 2. DAILY LESSON EMAIL
// ----------------------------------------------------------------------------
export function dailyLessonEmail(
  name: string,
  lessonTitle: string,
  lessonEmoji: string,
  lessonCategory: string,
  dayNumber: number,
  lessonUrl: string
): { subject: string; html: string; text: string } {
  const displayName = name || 'friend';
  const fact = getRandomFact();
  
  const html = emailWrapper(`
    ${emailHeader()}
    <tr>
      <td style="padding: 0 40px 30px;">
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">
          Good morning, ${displayName}! ☀️
        </p>
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 30px;">
          Your daily dose of curiosity is ready.
        </p>
        
        <!-- Lesson Card -->
        <table width="100%" style="background-color: ${BRAND_COLORS.border}; border-radius: 12px; margin-bottom: 30px;">
          <tr>
            <td style="padding: 24px;">
              <p style="color: ${BRAND_COLORS.textDim}; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 8px;">
                Day ${dayNumber} · ${lessonCategory}
              </p>
              <h2 style="color: ${BRAND_COLORS.text}; font-size: 24px; font-weight: 600; margin: 0 0 16px;">
                ${lessonEmoji} ${lessonTitle}
              </h2>
              <p style="color: ${BRAND_COLORS.textMuted}; font-size: 14px; margin: 0;">
                ⏱️ 5 minutes · 🎯 Perfect for your morning
              </p>
            </td>
          </tr>
        </table>
        
        ${ctaButton("Start Today's Lesson →", lessonUrl)}
        
        <p style="color: ${BRAND_COLORS.text}; font-size: 16px; line-height: 1.6; margin: 20px 0 0;">
          ✨ Stay curious,<br><strong>Kelly</strong>
        </p>
      </td>
    </tr>
    ${funFactBox(fact)}
    ${emailFooter()}
  `);

  const text = `
Good morning, ${displayName}! ☀️

Your daily dose of curiosity is ready.

═══════════════════════════════
Day ${dayNumber} · ${lessonCategory}

${lessonEmoji} ${lessonTitle}

⏱️ 5 minutes · 🎯 Perfect for your morning
═══════════════════════════════

Start today's lesson: ${lessonUrl}

✨ Stay curious,
Kelly

---
💡 Did you know? ${fact}

---
✨ Curious Kelly | Learn something new every day
curiouskelly.com | © 2025 Lesson of the Day PBC
  `.trim();

  return {
    subject: `${lessonEmoji} Today's lesson: ${lessonTitle}`,
    html,
    text,
  };
}

// ----------------------------------------------------------------------------
// 3. STREAK CELEBRATION EMAIL
// ----------------------------------------------------------------------------
export function streakEmail(
  name: string,
  streakDays: number
): { subject: string; html: string; text: string } {
  const displayName = name || 'Curious Friend';
  const fact = getRandomFact();
  
  // Milestone messages
  const milestoneMessages: Record<number, { emoji: string; title: string; message: string }> = {
    3: { emoji: '🔥', title: '3-Day Streak!', message: "You're building momentum! Three days of learning is how habits start." },
    7: { emoji: '⭐', title: 'One Week Wonder!', message: "A full week of curiosity! You're officially forming a learning habit." },
    14: { emoji: '🌟', title: 'Two Week Champion!', message: "Two weeks straight! Your brain is literally growing new connections." },
    30: { emoji: '🏆', title: 'Monthly Master!', message: "30 days of daily learning! You're in the top 5% of curious minds." },
    60: { emoji: '💎', title: 'Diamond Dedication!', message: "60 days! You've proven that curiosity is a lifestyle, not a phase." },
    90: { emoji: '🌈', title: 'Quarter Century of Curiosity!', message: "90 days of learning! You're an inspiration to curious minds everywhere." },
    100: { emoji: '💯', title: '100 Days of Wonder!', message: "ONE HUNDRED DAYS! You're a true learning legend." },
    365: { emoji: '🎉', title: 'ONE YEAR OF CURIOSITY!', message: "365 days of learning something new. You're extraordinary!" },
  };
  
  const milestone = milestoneMessages[streakDays] || {
    emoji: '🔥',
    title: `${streakDays}-Day Streak!`,
    message: `${streakDays} days of curiosity! Keep that momentum going!`
  };

  const html = emailWrapper(`
    ${emailHeader()}
    <tr>
      <td style="padding: 0 40px 30px; text-align: center;">
        <p style="font-size: 64px; margin: 0 0 20px;">${milestone.emoji}</p>
        <h1 style="color: ${BRAND_COLORS.text}; font-size: 32px; font-weight: 700; margin: 0 0 16px;">
          ${milestone.title}
        </h1>
        <p style="color: ${BRAND_COLORS.gold}; font-size: 48px; font-weight: 700; margin: 0 0 20px;">
          ${streakDays} Days
        </p>
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 18px; line-height: 1.6; margin: 0 0 30px;">
          ${milestone.message}
        </p>
        ${ctaButton("Keep the Streak Alive →", 'https://curiouskelly.com/learn', BRAND_COLORS.gold)}
        <p style="color: ${BRAND_COLORS.text}; font-size: 16px; line-height: 1.6; margin: 30px 0 0;">
          So proud of you, ${displayName}! ✨<br><strong>Kelly</strong>
        </p>
      </td>
    </tr>
    ${funFactBox(fact)}
    ${emailFooter()}
  `);

  const text = `
${milestone.emoji} ${milestone.title}

${streakDays} DAYS!

${milestone.message}

Keep the streak alive: https://curiouskelly.com/learn

So proud of you, ${displayName}! ✨
Kelly

---
💡 Did you know? ${fact}

---
✨ Curious Kelly | Learn something new every day
curiouskelly.com | © 2025 Lesson of the Day PBC
  `.trim();

  return {
    subject: `${milestone.emoji} ${milestone.title} You're on fire!`,
    html,
    text,
  };
}

// ----------------------------------------------------------------------------
// 4. AFFILIATE WELCOME EMAIL
// ----------------------------------------------------------------------------
export function affiliateWelcomeEmail(
  name: string,
  affiliateCode: string,
  affiliateUrl: string
): { subject: string; html: string; text: string } {
  const displayName = name || 'Partner';
  
  const html = emailWrapper(`
    ${emailHeader(`Welcome to the Kelly Affiliate Family! 🤝`)}
    <tr>
      <td style="padding: 0 40px 30px;">
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">
          Hey ${displayName}!
        </p>
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">
          I'm thrilled to have you as a Curious Kelly partner! Together, we're going to help more people discover the joy of learning something new every day.
        </p>
        
        <!-- Affiliate Code Box -->
        <table width="100%" style="background-color: ${BRAND_COLORS.border}; border-radius: 12px; margin: 30px 0;">
          <tr>
            <td style="padding: 24px; text-align: center;">
              <p style="color: ${BRAND_COLORS.textDim}; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 12px;">
                Your Affiliate Code
              </p>
              <p style="color: ${BRAND_COLORS.gold}; font-size: 32px; font-weight: 700; font-family: monospace; margin: 0 0 16px;">
                ${affiliateCode}
              </p>
              <p style="color: ${BRAND_COLORS.textMuted}; font-size: 14px; margin: 0;">
                Your referral link:<br>
                <a href="${affiliateUrl}" style="color: ${BRAND_COLORS.accent}; word-break: break-all;">${affiliateUrl}</a>
              </p>
            </td>
          </tr>
        </table>
        
        <p style="color: ${BRAND_COLORS.text}; font-size: 16px; line-height: 1.6; margin: 0 0 10px; font-weight: 600;">
          Here's what you earn:
        </p>
        <ul style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.8; margin: 0 0 20px; padding-left: 20px;">
          <li><strong style="color: ${BRAND_COLORS.gold};">30% commission</strong> on every subscription</li>
          <li>Recurring revenue for as long as they stay subscribed</li>
          <li>Payouts every month via PayPal or bank transfer</li>
          <li>Real-time dashboard to track your referrals</li>
        </ul>
        
        ${ctaButton('Access Your Affiliate Dashboard →', 'https://curiouskelly.com/affiliate/dashboard')}
        
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 20px 0 0;">
          Need marketing materials? We've got you covered with ready-to-use social posts, images, and talking points.
        </p>
        
        <p style="color: ${BRAND_COLORS.text}; font-size: 16px; line-height: 1.6; margin: 20px 0 0;">
          ✨ Let's grow together,<br><strong>Kelly</strong>
        </p>
        <p style="color: ${BRAND_COLORS.textDim}; font-size: 14px; margin: 20px 0 0;">
          Questions? Reply to this email anytime!
        </p>
      </td>
    </tr>
    ${emailFooter()}
  `);

  const text = `
Welcome to the Kelly Affiliate Family! 🤝

Hey ${displayName}!

I'm thrilled to have you as a Curious Kelly partner! Together, we're going to help more people discover the joy of learning something new every day.

═══════════════════════════════
YOUR AFFILIATE CODE: ${affiliateCode}

Your referral link:
${affiliateUrl}
═══════════════════════════════

Here's what you earn:
• 30% commission on every subscription
• Recurring revenue for as long as they stay subscribed
• Payouts every month via PayPal or bank transfer
• Real-time dashboard to track your referrals

Access your dashboard: https://curiouskelly.com/affiliate/dashboard

Need marketing materials? We've got you covered with ready-to-use social posts, images, and talking points.

✨ Let's grow together,
Kelly

Questions? Reply to this email anytime!

---
✨ Curious Kelly | Learn something new every day
curiouskelly.com | © 2025 Lesson of the Day PBC
  `.trim();

  return {
    subject: `🤝 Welcome to the Curious Kelly Affiliate Program!`,
    html,
    text,
  };
}

// ----------------------------------------------------------------------------
// 5. AFFILIATE PAYOUT EMAIL
// ----------------------------------------------------------------------------
export function affiliatePayoutEmail(
  name: string,
  amount: string,
  referralCount: number,
  payoutMethod: string,
  period: string
): { subject: string; html: string; text: string } {
  const displayName = name || 'Partner';
  
  const html = emailWrapper(`
    ${emailHeader()}
    <tr>
      <td style="padding: 0 40px 30px; text-align: center;">
        <p style="font-size: 48px; margin: 0 0 16px;">💰</p>
        <h1 style="color: ${BRAND_COLORS.text}; font-size: 28px; font-weight: 600; margin: 0 0 8px;">
          Payout on the way!
        </h1>
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; margin: 0 0 30px;">
          Great work, ${displayName}!
        </p>
        
        <!-- Payout Details -->
        <table width="100%" style="background-color: ${BRAND_COLORS.border}; border-radius: 12px; margin: 0 0 30px;">
          <tr>
            <td style="padding: 24px; text-align: center;">
              <p style="color: ${BRAND_COLORS.gold}; font-size: 48px; font-weight: 700; margin: 0 0 8px;">
                ${amount}
              </p>
              <p style="color: ${BRAND_COLORS.textMuted}; font-size: 14px; margin: 0;">
                From ${referralCount} referral${referralCount !== 1 ? 's' : ''} · ${period}
              </p>
            </td>
          </tr>
        </table>
        
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">
          Your payout is being sent via <strong style="color: ${BRAND_COLORS.text};">${payoutMethod}</strong> and should arrive within 2-3 business days.
        </p>
        
        ${ctaButton('View Payout Details →', 'https://curiouskelly.com/affiliate/payouts')}
        
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 30px 0 0;">
          Thank you for spreading curiosity! 🙏
        </p>
        <p style="color: ${BRAND_COLORS.text}; font-size: 16px; line-height: 1.6; margin: 20px 0 0;">
          ✨ Gratefully,<br><strong>Kelly</strong>
        </p>
      </td>
    </tr>
    ${emailFooter()}
  `);

  const text = `
💰 Payout on the way!

Great work, ${displayName}!

═══════════════════════════════
PAYOUT AMOUNT: ${amount}
From ${referralCount} referral${referralCount !== 1 ? 's' : ''} · ${period}
═══════════════════════════════

Your payout is being sent via ${payoutMethod} and should arrive within 2-3 business days.

View payout details: https://curiouskelly.com/affiliate/payouts

Thank you for spreading curiosity! 🙏

✨ Gratefully,
Kelly

---
✨ Curious Kelly | Learn something new every day
curiouskelly.com | © 2025 Lesson of the Day PBC
  `.trim();

  return {
    subject: `💰 Your ${amount} affiliate payout is on the way!`,
    html,
    text,
  };
}

// ----------------------------------------------------------------------------
// 6. RE-ENGAGEMENT EMAIL (Miss You!)
// ----------------------------------------------------------------------------
export function reEngagementEmail(
  name: string,
  daysMissed: number,
  lastLessonTitle: string
): { subject: string; html: string; text: string } {
  const displayName = name || 'friend';
  const fact = getRandomFact();
  
  const html = emailWrapper(`
    ${emailHeader()}
    <tr>
      <td style="padding: 0 40px 30px;">
        <p style="font-size: 48px; text-align: center; margin: 0 0 20px;">🥺</p>
        <h1 style="color: ${BRAND_COLORS.text}; font-size: 28px; font-weight: 600; margin: 0 0 20px; text-align: center;">
          I miss you, ${displayName}!
        </h1>
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">
          It's been ${daysMissed} days since our last lesson together. Your curious mind has been quiet, and I've been saving up so many interesting things to share!
        </p>
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">
          Last time, we explored <strong style="color: ${BRAND_COLORS.text};">"${lastLessonTitle}"</strong> together. Ready to pick up where we left off?
        </p>
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 30px;">
          No pressure, no guilt - just whenever you're ready. Learning should feel like a gift, not a chore. ❤️
        </p>
        
        ${ctaButton("Let's Learn Together →", 'https://curiouskelly.com/learn')}
        
        <p style="color: ${BRAND_COLORS.text}; font-size: 16px; line-height: 1.6; margin: 20px 0 0;">
          ✨ Always here for you,<br><strong>Kelly</strong>
        </p>
      </td>
    </tr>
    ${funFactBox(fact)}
    ${emailFooter()}
  `);

  const text = `
🥺 I miss you, ${displayName}!

It's been ${daysMissed} days since our last lesson together. Your curious mind has been quiet, and I've been saving up so many interesting things to share!

Last time, we explored "${lastLessonTitle}" together. Ready to pick up where we left off?

No pressure, no guilt - just whenever you're ready. Learning should feel like a gift, not a chore. ❤️

Let's learn together: https://curiouskelly.com/learn

✨ Always here for you,
Kelly

---
💡 Did you know? ${fact}

---
✨ Curious Kelly | Learn something new every day
curiouskelly.com | © 2025 Lesson of the Day PBC
  `.trim();

  return {
    subject: `🥺 ${displayName}, I miss our learning adventures!`,
    html,
    text,
  };
}

// ----------------------------------------------------------------------------
// 7. PASSWORD RESET EMAIL
// ----------------------------------------------------------------------------
export function passwordResetEmail(
  name: string,
  resetUrl: string
): { subject: string; html: string; text: string } {
  const displayName = name || 'friend';
  
  const html = emailWrapper(`
    ${emailHeader()}
    <tr>
      <td style="padding: 0 40px 30px;">
        <h1 style="color: ${BRAND_COLORS.text}; font-size: 28px; font-weight: 600; margin: 0 0 20px; text-align: center;">
          Reset your password 🔐
        </h1>
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 20px;">
          Hey ${displayName},
        </p>
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 30px;">
          We received a request to reset your password. Click the button below to create a new one. This link expires in 1 hour.
        </p>
        
        ${ctaButton('Reset Password →', resetUrl)}
        
        <p style="color: ${BRAND_COLORS.textDim}; font-size: 14px; line-height: 1.6; margin: 30px 0 0;">
          If you didn't request this, you can safely ignore this email. Your password won't change unless you click the link above.
        </p>
        <p style="color: ${BRAND_COLORS.text}; font-size: 16px; line-height: 1.6; margin: 20px 0 0;">
          ✨ Stay curious,<br><strong>Kelly</strong>
        </p>
      </td>
    </tr>
    ${emailFooter()}
  `);

  const text = `
Reset your password 🔐

Hey ${displayName},

We received a request to reset your password. Click the link below to create a new one. This link expires in 1 hour.

Reset password: ${resetUrl}

If you didn't request this, you can safely ignore this email. Your password won't change unless you click the link above.

✨ Stay curious,
Kelly

---
✨ Curious Kelly | Learn something new every day
curiouskelly.com | © 2025 Lesson of the Day PBC
  `.trim();

  return {
    subject: `🔐 Reset your Curious Kelly password`,
    html,
    text,
  };
}

// ----------------------------------------------------------------------------
// 8. SUBSCRIPTION CONFIRMATION EMAIL
// ----------------------------------------------------------------------------
export function subscriptionConfirmEmail(
  name: string,
  plan: string,
  amount: string,
  nextBillingDate: string
): { subject: string; html: string; text: string } {
  const displayName = name || 'Curious Friend';
  
  const html = emailWrapper(`
    ${emailHeader()}
    <tr>
      <td style="padding: 0 40px 30px; text-align: center;">
        <p style="font-size: 48px; margin: 0 0 16px;">🎉</p>
        <h1 style="color: ${BRAND_COLORS.text}; font-size: 28px; font-weight: 600; margin: 0 0 20px;">
          You're all set, ${displayName}!
        </h1>
        <p style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.6; margin: 0 0 30px;">
          Thank you for subscribing to Curious Kelly! You now have unlimited access to every lesson, past and future.
        </p>
        
        <!-- Plan Details -->
        <table width="100%" style="background-color: ${BRAND_COLORS.border}; border-radius: 12px; margin: 0 0 30px;">
          <tr>
            <td style="padding: 24px;">
              <p style="color: ${BRAND_COLORS.textDim}; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 8px;">
                Your Plan
              </p>
              <p style="color: ${BRAND_COLORS.text}; font-size: 24px; font-weight: 600; margin: 0 0 16px;">
                ${plan}
              </p>
              <p style="color: ${BRAND_COLORS.textMuted}; font-size: 14px; margin: 0;">
                ${amount} · Next billing: ${nextBillingDate}
              </p>
            </td>
          </tr>
        </table>
        
        <p style="color: ${BRAND_COLORS.text}; font-size: 16px; line-height: 1.6; margin: 0 0 10px; font-weight: 600;">
          What you can do now:
        </p>
        <ul style="color: ${BRAND_COLORS.textMuted}; font-size: 16px; line-height: 1.8; margin: 0 0 30px; padding-left: 20px; text-align: left;">
          <li>Access all 365 daily lessons</li>
          <li>Learn at your own pace</li>
          <li>Track your progress and streaks</li>
          <li>Get personalized recommendations</li>
        </ul>
        
        ${ctaButton('Start Exploring →', 'https://curiouskelly.com/learn')}
        
        <p style="color: ${BRAND_COLORS.text}; font-size: 16px; line-height: 1.6; margin: 20px 0 0;">
          ✨ Thank you for believing in curiosity,<br><strong>Kelly</strong>
        </p>
      </td>
    </tr>
    ${emailFooter()}
  `);

  const text = `
🎉 You're all set, ${displayName}!

Thank you for subscribing to Curious Kelly! You now have unlimited access to every lesson, past and future.

═══════════════════════════════
YOUR PLAN: ${plan}
${amount} · Next billing: ${nextBillingDate}
═══════════════════════════════

What you can do now:
• Access all 365 daily lessons
• Learn at your own pace
• Track your progress and streaks
• Get personalized recommendations

Start exploring: https://curiouskelly.com/learn

✨ Thank you for believing in curiosity,
Kelly

---
✨ Curious Kelly | Learn something new every day
curiouskelly.com | © 2025 Lesson of the Day PBC
  `.trim();

  return {
    subject: `🎉 Welcome to Curious Kelly ${plan}!`,
    html,
    text,
  };
}

