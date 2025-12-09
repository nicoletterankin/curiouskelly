/**
 * Contact Form & Document Submission API
 * 
 * Handles contact form submissions and document uploads.
 * Forwards all submissions to hello@curiouskelly.com via Resend.
 * 
 * Supports:
 * - General inquiries
 * - Document/file submissions (base64 encoded)
 * - Artist asset uploads (for character references, etc.)
 * 
 * Environment Variables Required:
 * - RESEND_API_KEY: Your Resend API key
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';

const RESEND_API_URL = 'https://api.resend.com/emails';

// Maximum attachment size (5MB in base64)
const MAX_ATTACHMENT_SIZE = 5 * 1024 * 1024;

interface ContactFormData {
  name: string;
  email: string;
  subject?: string;
  message: string;
  source?: string; // Which page the form was submitted from
  attachments?: Array<{
    filename: string;
    content: string; // base64 encoded
    contentType?: string;
  }>;
}

function generateContactEmailHTML(data: ContactFormData): string {
  const attachmentInfo = data.attachments?.length 
    ? `<p style="color: #6b7280; font-size: 14px; margin-top: 20px; padding-top: 20px; border-top: 1px solid #e5e7eb;">
        📎 ${data.attachments.length} attachment(s) included
       </p>`
    : '';

  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1f2937; max-width: 600px; margin: 0 auto; padding: 20px;">
  <div style="background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%); padding: 24px; border-radius: 12px 12px 0 0;">
    <h1 style="color: white; margin: 0; font-size: 20px;">✨ New Contact Form Submission</h1>
  </div>
  
  <div style="background: #f9fafb; padding: 24px; border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 12px 12px;">
    <table style="width: 100%; border-collapse: collapse;">
      <tr>
        <td style="padding: 8px 0; color: #6b7280; font-size: 14px; width: 100px;">From:</td>
        <td style="padding: 8px 0; font-weight: 600;">${escapeHtml(data.name)}</td>
      </tr>
      <tr>
        <td style="padding: 8px 0; color: #6b7280; font-size: 14px;">Email:</td>
        <td style="padding: 8px 0;"><a href="mailto:${escapeHtml(data.email)}" style="color: #3b82f6;">${escapeHtml(data.email)}</a></td>
      </tr>
      <tr>
        <td style="padding: 8px 0; color: #6b7280; font-size: 14px;">Source:</td>
        <td style="padding: 8px 0; color: #6b7280; font-size: 14px;">${escapeHtml(data.source || 'Unknown')}</td>
      </tr>
    </table>
    
    <div style="margin-top: 24px; padding: 20px; background: white; border-radius: 8px; border: 1px solid #e5e7eb;">
      <h3 style="margin: 0 0 12px 0; color: #1f2937; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em;">Message</h3>
      <p style="margin: 0; white-space: pre-wrap;">${escapeHtml(data.message)}</p>
    </div>
    
    ${attachmentInfo}
  </div>
  
  <p style="text-align: center; color: #9ca3af; font-size: 12px; margin-top: 24px;">
    Curious Kelly Contact System • ${new Date().toISOString().split('T')[0]}
  </p>
</body>
</html>
  `.trim();
}

function generateContactEmailText(data: ContactFormData): string {
  const attachmentInfo = data.attachments?.length 
    ? `\n\n---\n📎 ${data.attachments.length} attachment(s) included`
    : '';

  return `
NEW CONTACT FORM SUBMISSION
============================

From: ${data.name}
Email: ${data.email}
Source: ${data.source || 'Unknown'}

MESSAGE:
${data.message}
${attachmentInfo}

---
Curious Kelly Contact System
${new Date().toISOString()}
  `.trim();
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers for browser requests
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle preflight
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Get request body
  const { name, email, subject, message, source, attachments } = req.body as ContactFormData;

  // Validate required fields
  if (!name || !email || !message) {
    return res.status(400).json({ 
      error: 'Missing required fields',
      required: ['name', 'email', 'message']
    });
  }

  // Validate email format
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    return res.status(400).json({ error: 'Invalid email format' });
  }

  // Validate attachments if present
  if (attachments && attachments.length > 0) {
    for (const attachment of attachments) {
      if (!attachment.filename || !attachment.content) {
        return res.status(400).json({ error: 'Invalid attachment format' });
      }
      // Check size (base64 adds ~33% overhead)
      if (attachment.content.length > MAX_ATTACHMENT_SIZE * 1.34) {
        return res.status(400).json({ 
          error: `Attachment "${attachment.filename}" exceeds 5MB limit` 
        });
      }
    }
  }

  // Get Resend API key
  const resendApiKey = process.env.RESEND_API_KEY;
  
  if (!resendApiKey) {
    console.error('RESEND_API_KEY not configured');
    return res.status(500).json({ error: 'Email service not configured' });
  }

  try {
    // Build email payload
    const emailPayload: Record<string, unknown> = {
      from: 'Kelly <hello@curiouskelly.com>',
      to: 'hello@curiouskelly.com',
      reply_to: email,
      subject: subject || `Contact Form: ${name}`,
      html: generateContactEmailHTML({ name, email, subject, message, source, attachments }),
      text: generateContactEmailText({ name, email, subject, message, source, attachments }),
      tags: [
        { name: 'type', value: 'contact_form' },
        { name: 'source', value: source || 'unknown' }
      ]
    };

    // Add attachments if present
    if (attachments && attachments.length > 0) {
      emailPayload.attachments = attachments.map(att => ({
        filename: att.filename,
        content: att.content,
        content_type: att.contentType || 'application/octet-stream'
      }));
    }

    const response = await fetch(RESEND_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${resendApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(emailPayload),
    });

    const data = await response.json();

    if (!response.ok) {
      console.error('Resend API error:', data);
      return res.status(response.status).json({ 
        error: 'Failed to send message',
        details: data 
      });
    }

    console.log(`Contact form submitted from ${email}`, { 
      id: data.id, 
      source, 
      hasAttachments: attachments?.length || 0 
    });

    // Send confirmation email to the sender
    try {
      await fetch(RESEND_API_URL, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${resendApiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          from: 'Kelly <hello@curiouskelly.com>',
          to: email,
          subject: "Thanks for reaching out! ✨",
          html: `
            <p style="font-family: Georgia, serif; font-size: 18px; color: #1f2937; line-height: 1.8; max-width: 480px;">
              Hi ${escapeHtml(name)},<br><br>
              Thanks for your message! I received it and will get back to you soon.<br><br>
              ${attachments?.length ? `I also received your ${attachments.length} file(s) — thank you for sending those!<br><br>` : ''}
              Stay curious,<br>
              <span style="color: #6b7280;">— Kelly</span>
            </p>
          `,
          text: `Hi ${name},\n\nThanks for your message! I received it and will get back to you soon.\n\n${attachments?.length ? `I also received your ${attachments.length} file(s) — thank you for sending those!\n\n` : ''}Stay curious,\n— Kelly`,
          tags: [{ name: 'type', value: 'contact_confirmation' }]
        }),
      });
    } catch (confirmError) {
      // Don't fail if confirmation email fails
      console.warn('Failed to send confirmation email:', confirmError);
    }
    
    return res.status(200).json({ 
      success: true,
      message: 'Message sent successfully! Check your email for confirmation.',
      id: data.id 
    });

  } catch (error) {
    console.error('Error processing contact form:', error);
    return res.status(500).json({ 
      error: 'Failed to process submission',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}





