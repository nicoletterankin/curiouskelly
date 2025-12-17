#!/usr/bin/env npx tsx
/**
 * Send Test Email - Day 351
 * 
 * Sends the Day 351 summary email directly using Resend API
 * Usage: npx tsx scripts/send-test-email.ts nicoletterankin@gmail.com
 */

import 'dotenv/config';
import * as fs from 'fs';
import * as path from 'path';

const RESEND_API_URL = 'https://api.resend.com/emails';

async function sendTestEmail(toEmail: string) {
  const resendApiKey = process.env.RESEND_API_KEY;
  
  if (!resendApiKey) {
    console.error('❌ RESEND_API_KEY not found in environment');
    console.log('   Make sure your .env file has RESEND_API_KEY=re_...');
    process.exit(1);
  }

  // Read the email HTML
  const emailPath = path.join(process.cwd(), 'generated-emails', 'day-351-email.html');
  if (!fs.existsSync(emailPath)) {
    console.error('❌ Email HTML not found at:', emailPath);
    process.exit(1);
  }

  const html = fs.readFileSync(emailPath, 'utf-8');

  // Generate plain text version
  const text = `
Good morning.

🔮 Today's Lesson: Practicing in Your Mind
December 17 · Meta-Learning

Your brain can't tell the difference between doing and imagining.

Ever wondered why athletes close their eyes before a big moment? They're not just calming their nerves. They're doing something far more powerful—they're practicing. Without moving a muscle.

It's called visualization, and the science behind it might change how you think about learning itself.

💡 What you'll discover:
• Brain scans show visualization activates 90% of the same neural areas as actually doing something
• Olympic athletes spend up to 50% of their training time on mental rehearsal
• Pianists who only visualized practicing improved nearly as much as those who physically practiced

Five minutes. I think you'll love it.

▶ Watch the lesson: https://curiouskelly.com/videos/summary/day-351.mp4

🎯 Today's Growth Challenge: Learning Accountability
Choose one person you trust and tell them about your learning goal for the week. Ask them to check in with you in 7 days.

✨ Today's Wisdom
"The mind that rehearses grows stronger than the mind that merely waits."

— Kelly

---
December 17 · curiouskelly.com
Lesson of the Day PBC · hello@curiouskelly.com
  `.trim();

  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  📧 SENDING TEST EMAIL - DAY 351                               ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  console.log('');
  console.log(`   To: ${toEmail}`);
  console.log(`   Subject: 🔮 Today's lesson: Practicing in Your Mind`);
  console.log('');

  try {
    const response = await fetch(RESEND_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${resendApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'Kelly <hello@curiouskelly.com>',
        to: toEmail,
        subject: "🔮 Today's lesson: Practicing in Your Mind",
        html,
        text,
        reply_to: 'hello@curiouskelly.com',
      }),
    });

    const data = await response.json();
    
    if (!response.ok) {
      console.error('❌ Failed to send email:', data);
      process.exit(1);
    }

    console.log('✅ EMAIL SENT SUCCESSFULLY!');
    console.log('');
    console.log(`   Resend ID: ${data.id}`);
    console.log('');
    console.log('   📬 Check your inbox (and spam folder)!');
    console.log('');

  } catch (error) {
    console.error('❌ Error sending email:', error);
    process.exit(1);
  }
}

// Get email from command line
const email = process.argv[2];
if (!email || !email.includes('@')) {
  console.log('Usage: npx tsx scripts/send-test-email.ts your@email.com');
  process.exit(1);
}

sendTestEmail(email);
