#!/usr/bin/env npx tsx
/**
 * Test Subscribe Flow
 * 
 * Simulates a user subscribing via the subscribe page
 * Usage: npx tsx scripts/test-subscribe.ts nicoletterankin@gmail.com "Nicolette"
 */

import 'dotenv/config';

const RESEND_API_URL = 'https://api.resend.com/emails';

async function testSubscribe(email: string, name: string) {
  const resendApiKey = process.env.RESEND_API_KEY;
  
  if (!resendApiKey) {
    console.error('❌ RESEND_API_KEY not found');
    process.exit(1);
  }

  console.log('');
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║  🎯 TESTING SUBSCRIPTION FLOW                                  ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  console.log('');
  console.log(`   Name: ${name}`);
  console.log(`   Email: ${email}`);
  console.log('');

  const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; background: #fafafa; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background: #fafafa;">
    <tr>
      <td align="center" style="padding: 40px 20px;">
        <table width="100%" style="max-width: 480px;">
          <tr>
            <td style="text-align: center; padding-bottom: 24px;">
              <img src="https://curiouskelly.com/images/brand/kelly-mark-circle-128.png" alt="Kelly" width="80" height="80" style="border-radius: 50%; border: 3px solid #3b82f6;">
            </td>
          </tr>
          <tr>
            <td>
              <p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; margin: 0 0 20px;">
                Hi ${name} — I'm Kelly.
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 17px; color: #374151; line-height: 1.9; margin: 0 0 20px;">
                I don't have all the answers. But I love finding them. And I think learning is better together.
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 17px; color: #374151; line-height: 1.9; margin: 0 0 20px;">
                Starting tomorrow, you'll get a daily email with:
              </p>
              
              <ul style="font-family: Georgia, serif; font-size: 16px; color: #374151; line-height: 1.9; margin: 0 0 20px; padding-left: 20px;">
                <li>🔮 <strong>Learn</strong> — A fascinating topic in 5 minutes</li>
                <li>🎯 <strong>Grow</strong> — A practical challenge to build skills</li>
                <li>🎬 <strong>Watch</strong> — A 100-second video summary</li>
              </ul>
              
              <p style="font-family: Georgia, serif; font-size: 17px; color: #374151; line-height: 1.9; margin: 0 0 24px;">
                But why wait? Today's lesson is ready now.
              </p>
              
              <p style="text-align: center; margin: 0 0 24px;">
                <a href="https://curiouskelly.com/watch/day-351.html" style="display: inline-block; background: #3b82f6; color: white; padding: 14px 28px; border-radius: 8px; text-decoration: none; font-family: -apple-system, sans-serif; font-size: 16px; font-weight: 600;">
                  ▶️ Watch today's lesson (100 sec) →
                </a>
              </p>
              
              <p style="font-family: Georgia, serif; font-size: 15px; color: #6b7280; font-style: italic; margin: 0;">
                — Kelly
              </p>
            </td>
          </tr>
          <tr>
            <td style="padding-top: 32px; border-top: 1px solid #e5e7eb; margin-top: 32px;">
              <p style="font-family: -apple-system, sans-serif; font-size: 12px; color: #9ca3af; text-align: center; margin: 0;">
                ✨ Curious Kelly · <a href="https://curiouskelly.com" style="color: #9ca3af;">curiouskelly.com</a><br>
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
  `.trim();

  const text = `
Hi ${name} — I'm Kelly.

I don't have all the answers. But I love finding them. And I think learning is better together.

Starting tomorrow, you'll get a daily email with:
- 🔮 Learn — A fascinating topic in 5 minutes
- 🎯 Grow — A practical challenge to build skills
- 🎬 Watch — A 100-second video summary

But why wait? Today's lesson is ready now:
▶️ Watch: https://curiouskelly.com/watch/day-351.html

— Kelly

---
✨ Curious Kelly · curiouskelly.com
Lesson of the Day PBC · hello@curiouskelly.com
  `.trim();

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
        subject: "Welcome! Your daily lessons start tomorrow ✨",
        html,
        text,
        reply_to: 'hello@curiouskelly.com',
      }),
    });

    const data = await response.json();
    
    console.log('📤 Resend Response:');
    console.log(JSON.stringify(data, null, 2));
    console.log('');
    
    if (!response.ok) {
      console.error('❌ Failed to send email');
      console.error('   Status:', response.status);
      process.exit(1);
    }

    console.log('✅ WELCOME EMAIL SENT!');
    console.log('');
    console.log(`   Resend ID: ${data.id}`);
    console.log('');
    console.log('   📬 Check your inbox!');
    console.log('   📁 Also check spam/promotions folder');
    console.log('');

  } catch (error) {
    console.error('❌ Error:', error);
    process.exit(1);
  }
}

// Get args
const email = process.argv[2];
const name = process.argv[3] || 'Friend';

if (!email || !email.includes('@')) {
  console.log('Usage: npx tsx scripts/test-subscribe.ts your@email.com "Your Name"');
  process.exit(1);
}

testSubscribe(email, name);
