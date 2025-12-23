/**
 * Contact Form & Document Submission API
 * 
 * Handles contact form submissions and document uploads.
 * - Uploads files to Supabase Storage (submissions bucket)
 * - Saves submission metadata to database
 * - Forwards notification to hello@curiouskelly.com via Resend
 * - Sends confirmation email to sender
 * 
 * Supports:
 * - General inquiries
 * - Document/file submissions (up to 50MB each)
 * - Artist asset uploads (FBX, OBJ, Blend, etc.)
 * 
 * Environment Variables Required:
 * - RESEND_API_KEY: Your Resend API key
 * - PUBLIC_SUPABASE_URL or NEXT_PUBLIC_SUPABASE_URL: Supabase project URL
 * - SUPABASE_SERVICE_ROLE_KEY: Service role key for storage uploads
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';
import type { SupabaseClient } from '@supabase/supabase-js';

type AnySupabaseClient = SupabaseClient<any, any, any, any, any>;

const RESEND_API_URL = 'https://api.resend.com/emails';

// Maximum attachment size (50MB)
const MAX_ATTACHMENT_SIZE = 50 * 1024 * 1024;

// Initialize Supabase client with service role
const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

interface Attachment {
  filename: string;
  content: string; // base64 encoded
  contentType?: string;
}

interface StoredAttachment {
  filename: string;
  storage_path: string;
  public_url: string;
  size_bytes: number;
  mime_type: string;
}

interface ContactFormData {
  name: string;
  email: string;
  subject?: string;
  message: string;
  source?: string;
  attachments?: Attachment[];
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function generateNotificationEmailHTML(
  data: ContactFormData, 
  storedAttachments: StoredAttachment[],
  submissionId: string
): string {
  const responsePageUrl = `https://curiouskelly.com/submission/${submissionId}`;
  
  const attachmentRows = storedAttachments.map(att => `
    <tr>
      <td style="padding: 8px 12px; border-bottom: 1px solid #e5e7eb;">
        <a href="${att.public_url}" style="color: #3b82f6; text-decoration: none; font-weight: 500;">
          📎 ${escapeHtml(att.filename)}
        </a>
      </td>
      <td style="padding: 8px 12px; border-bottom: 1px solid #e5e7eb; color: #6b7280; font-size: 13px;">
        ${formatFileSize(att.size_bytes)}
      </td>
      <td style="padding: 8px 12px; border-bottom: 1px solid #e5e7eb;">
        <a href="${att.public_url}" style="color: #3b82f6; font-size: 13px;">Download →</a>
      </td>
    </tr>
  `).join('');

  const attachmentSection = storedAttachments.length > 0 ? `
    <div style="margin-top: 24px; padding: 20px; background: #f0f9ff; border-radius: 8px; border: 1px solid #bae6fd;">
      <h3 style="margin: 0 0 12px 0; color: #0369a1; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em;">
        📁 Uploaded Files (${storedAttachments.length})
      </h3>
      <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 6px; overflow: hidden;">
        <thead>
          <tr style="background: #f8fafc;">
            <th style="padding: 10px 12px; text-align: left; font-size: 12px; color: #64748b; border-bottom: 1px solid #e5e7eb;">File</th>
            <th style="padding: 10px 12px; text-align: left; font-size: 12px; color: #64748b; border-bottom: 1px solid #e5e7eb;">Size</th>
            <th style="padding: 10px 12px; text-align: left; font-size: 12px; color: #64748b; border-bottom: 1px solid #e5e7eb;">Action</th>
          </tr>
        </thead>
        <tbody>
          ${attachmentRows}
        </tbody>
      </table>
      <p style="margin: 12px 0 0 0; font-size: 12px; color: #64748b;">
        Files stored in Supabase Storage. Links valid indefinitely.
      </p>
    </div>
  ` : '';

  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1f2937; max-width: 600px; margin: 0 auto; padding: 20px;">
  <div style="background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%); padding: 24px; border-radius: 12px 12px 0 0;">
    <h1 style="color: white; margin: 0; font-size: 20px;">✨ New Submission Received</h1>
    <p style="color: rgba(255,255,255,0.8); margin: 8px 0 0 0; font-size: 14px;">
      From: ${escapeHtml(data.source || 'Contact Form')}
    </p>
  </div>
  
  <div style="background: #f9fafb; padding: 24px; border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 12px 12px;">
    <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
      <tr>
        <td style="padding: 8px 0; color: #6b7280; font-size: 14px; width: 80px; vertical-align: top;">From:</td>
        <td style="padding: 8px 0; font-weight: 600;">${escapeHtml(data.name)}</td>
      </tr>
      <tr>
        <td style="padding: 8px 0; color: #6b7280; font-size: 14px; vertical-align: top;">Email:</td>
        <td style="padding: 8px 0;"><a href="mailto:${escapeHtml(data.email)}" style="color: #3b82f6;">${escapeHtml(data.email)}</a></td>
      </tr>
      <tr>
        <td style="padding: 8px 0; color: #6b7280; font-size: 14px; vertical-align: top;">ID:</td>
        <td style="padding: 8px 0; font-family: monospace; font-size: 12px; color: #6b7280;">${submissionId}</td>
      </tr>
    </table>
    
    <div style="padding: 20px; background: white; border-radius: 8px; border: 1px solid #e5e7eb;">
      <h3 style="margin: 0 0 12px 0; color: #1f2937; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em;">Message</h3>
      <p style="margin: 0; white-space: pre-wrap; color: #374151;">${escapeHtml(data.message)}</p>
    </div>
    
    ${attachmentSection}
    
    <div style="margin-top: 24px; padding-top: 20px; border-top: 1px solid #e5e7eb; display: flex; gap: 12px; flex-wrap: wrap;">
      <a href="mailto:${escapeHtml(data.email)}?subject=Re: ${escapeHtml(data.subject || 'Your submission to Curious Kelly')}" 
         style="display: inline-block; background: #3b82f6; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600;">
        Reply to ${escapeHtml(data.name)} →
      </a>
      <a href="${responsePageUrl}" 
         style="display: inline-block; background: #d4a574; color: #0a0a0b; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600;">
        View Response Page ✨
      </a>
    </div>
    
    <div style="margin-top: 16px; padding: 12px 16px; background: #fffbeb; border-radius: 8px; border: 1px solid #fde68a;">
      <p style="margin: 0; font-size: 13px; color: #92400e;">
        💡 The learner received a link to view their submission and can add comments/files there. 
        <a href="${responsePageUrl}" style="color: #d97706;">See what they see →</a>
      </p>
    </div>
  </div>
  
  <p style="text-align: center; color: #9ca3af; font-size: 12px; margin-top: 24px;">
    Curious Kelly Contact System • ${new Date().toISOString().split('T')[0]}
  </p>
</body>
</html>
  `.trim();
}

function generateNotificationEmailText(
  data: ContactFormData, 
  storedAttachments: StoredAttachment[],
  submissionId: string
): string {
  const responsePageUrl = `https://curiouskelly.com/submission/${submissionId}`;
  
  const attachmentList = storedAttachments.length > 0
    ? '\n\nUPLOADED FILES:\n' + storedAttachments.map(att => 
        `- ${att.filename} (${formatFileSize(att.size_bytes)})\n  ${att.public_url}`
      ).join('\n')
    : '';

  return `
NEW SUBMISSION RECEIVED
========================

From: ${data.name}
Email: ${data.email}
Source: ${data.source || 'Contact Form'}
ID: ${submissionId}

MESSAGE:
${data.message}
${attachmentList}

---
Reply: mailto:${data.email}
View Response Page: ${responsePageUrl}
(This is what the learner sees - they can add comments and files here)

Curious Kelly Contact System
${new Date().toISOString()}
  `.trim();
}

async function uploadToSupabase(
  supabase: AnySupabaseClient,
  attachment: Attachment,
  submissionId: string
): Promise<StoredAttachment | null> {
  try {
    // Decode base64 to buffer
    const buffer = Buffer.from(attachment.content, 'base64');
    const sizeBytes = buffer.length;

    // Check size
    if (sizeBytes > MAX_ATTACHMENT_SIZE) {
      console.warn(`File ${attachment.filename} exceeds 50MB limit`);
      return null;
    }

    // Generate storage path: submissions/{submission_id}/{timestamp}_{filename}
    const timestamp = Date.now();
    const safeName = attachment.filename.replace(/[^a-zA-Z0-9.-]/g, '_');
    const storagePath = `${submissionId}/${timestamp}_${safeName}`;

    // Upload to Supabase Storage
    const { data, error } = await supabase.storage
      .from('submissions')
      .upload(storagePath, buffer, {
        contentType: attachment.contentType || 'application/octet-stream',
        cacheControl: '3600',
        upsert: false
      });

    if (error) {
      console.error(`Failed to upload ${attachment.filename}:`, error);
      return null;
    }

    // Get signed URL (valid for 1 year)
    const { data: urlData } = await supabase.storage
      .from('submissions')
      .createSignedUrl(storagePath, 60 * 60 * 24 * 365); // 1 year

    return {
      filename: attachment.filename,
      storage_path: storagePath,
      public_url: urlData?.signedUrl || '',
      size_bytes: sizeBytes,
      mime_type: attachment.contentType || 'application/octet-stream'
    };
  } catch (error) {
    console.error(`Error uploading ${attachment.filename}:`, error);
    return null;
  }
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

  // Get API keys
  const resendApiKey = process.env.RESEND_API_KEY;
  
  if (!resendApiKey) {
    console.error('RESEND_API_KEY not configured');
    return res.status(500).json({ error: 'Email service not configured' });
  }

  if (!supabaseUrl || !supabaseServiceKey) {
    console.error('Supabase not configured');
    return res.status(500).json({ error: 'Storage service not configured' });
  }

  // Initialize Supabase client
  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  try {
    // Generate submission ID
    const submissionId = crypto.randomUUID();

    // Upload attachments to Supabase Storage
    const storedAttachments: StoredAttachment[] = [];
    
    if (attachments && attachments.length > 0) {
      for (const attachment of attachments) {
        const stored = await uploadToSupabase(supabase, attachment, submissionId);
        if (stored) {
          storedAttachments.push(stored);
        }
      }
    }

    // Get request metadata
    const ipAddress = (req.headers['x-forwarded-for'] as string)?.split(',')[0] || 
                      req.headers['x-real-ip'] as string || 
                      'unknown';
    const userAgent = req.headers['user-agent'] || 'unknown';

    // Save submission to database
    const { error: dbError } = await supabase
      .from('contact_submissions')
      .insert({
        id: submissionId,
        name,
        email,
        subject: subject || null,
        message,
        source: source || null,
        attachments: storedAttachments,
        ip_address: ipAddress,
        user_agent: userAgent
      });

    if (dbError) {
      console.error('Failed to save submission:', dbError);
      // Continue anyway - email is more important
    }

    // Send notification email to hello@curiouskelly.com
    const notificationResponse = await fetch(RESEND_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${resendApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'Kelly <hello@curiouskelly.com>',
        to: 'hello@curiouskelly.com',
        reply_to: email,
        subject: `📬 ${subject || `New submission from ${name}`}`,
        html: generateNotificationEmailHTML({ name, email, subject, message, source }, storedAttachments, submissionId),
        text: generateNotificationEmailText({ name, email, subject, message, source }, storedAttachments, submissionId),
        tags: [
          { name: 'type', value: 'contact_submission' },
          { name: 'source', value: source || 'unknown' },
          { name: 'has_attachments', value: storedAttachments.length > 0 ? 'yes' : 'no' }
        ]
      }),
    });

    const notificationData = await notificationResponse.json();

    if (!notificationResponse.ok) {
      console.error('Resend API error:', notificationData);
      return res.status(notificationResponse.status).json({ 
        error: 'Failed to send notification',
        details: notificationData 
      });
    }

    // Update submission with Resend email ID
    await supabase
      .from('contact_submissions')
      .update({ resend_email_id: notificationData.id })
      .eq('id', submissionId);

    console.log(`Submission ${submissionId} from ${email}`, { 
      files: storedAttachments.length,
      source,
      emailId: notificationData.id
    });

    // Send confirmation email to the sender with link to view their submission
    try {
      const fileList = storedAttachments.length > 0 
        ? `\n\nI received your ${storedAttachments.length} file(s):\n${storedAttachments.map(a => `• ${a.filename}`).join('\n')}\n`
        : '';

      // Generate response page URL (clean path)
      const responseUrl = `https://curiouskelly.com/submission/${submissionId}`;

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
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: Georgia, 'Times New Roman', serif; background-color: #f9fafb;">
  <div style="max-width: 520px; margin: 0 auto; padding: 40px 20px;">
    <!-- Header -->
    <div style="text-align: center; margin-bottom: 32px;">
      <div style="font-size: 2.5rem; margin-bottom: 8px;">✨</div>
      <h1 style="font-size: 24px; font-weight: 400; color: #1f2937; margin: 0;">Hi ${escapeHtml(name)}!</h1>
    </div>
    
    <!-- Main content -->
    <div style="background: white; border-radius: 16px; padding: 32px; box-shadow: 0 4px 16px rgba(0,0,0,0.08);">
      <p style="font-size: 18px; color: #374151; line-height: 1.8; margin: 0 0 20px 0;">
        Thanks for your message! I received it and will get back to you soon.
      </p>
      
      ${storedAttachments.length > 0 ? `
      <div style="background: #f0f9ff; border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
        <p style="margin: 0; font-size: 15px; color: #0369a1;">
          📁 I also received your <strong>${storedAttachments.length} file(s)</strong> — thank you for sending those! They're safely stored and I'll review them.
        </p>
      </div>
      ` : ''}
      
      <!-- Response page CTA -->
      <div style="background: linear-gradient(135deg, rgba(212, 165, 116, 0.15) 0%, rgba(135, 206, 235, 0.1) 100%); border-radius: 12px; padding: 24px; margin-top: 24px; text-align: center;">
        <p style="margin: 0 0 16px 0; font-size: 15px; color: #4b5563;">
          Want to see exactly what I received, add more files, or leave a note?
        </p>
        <a href="${responseUrl}" style="display: inline-block; background: #d4a574; color: #0a0a0b; text-decoration: none; padding: 14px 28px; border-radius: 10px; font-weight: 600; font-size: 15px;">
          View Your Submission →
        </a>
        <p style="margin: 16px 0 0 0; font-size: 12px; color: #9ca3af;">
          Bookmark this link to check back anytime
        </p>
      </div>
    </div>
    
    <!-- Sign off -->
    <div style="text-align: center; margin-top: 32px;">
      <p style="font-size: 16px; color: #6b7280; margin: 0;">
        Stay curious,<br>
        <span style="color: #1f2937; font-weight: 500;">— Kelly</span>
      </p>
    </div>
    
    <!-- Footer -->
    <div style="text-align: center; margin-top: 40px; padding-top: 24px; border-top: 1px solid #e5e7eb;">
      <p style="font-size: 13px; color: #9ca3af; margin: 0;">
        ✨ Curious Kelly • <a href="https://curiouskelly.com" style="color: #d4a574;">curiouskelly.com</a>
      </p>
    </div>
  </div>
</body>
</html>
          `,
          text: `Hi ${name},\n\nThanks for your message! I received it and will get back to you soon.${fileList}\n\nView your submission and add updates anytime:\n${responseUrl}\n\nStay curious,\n— Kelly`,
          tags: [{ name: 'type', value: 'contact_confirmation' }]
        }),
      });

      // Mark confirmation as sent
      await supabase
        .from('contact_submissions')
        .update({ confirmation_sent: true })
        .eq('id', submissionId);

    } catch (confirmError) {
      console.warn('Failed to send confirmation email:', confirmError);
    }
    
    return res.status(200).json({ 
      success: true,
      message: 'Message sent successfully! Check your email for confirmation.',
      id: submissionId,
      filesUploaded: storedAttachments.length
    });

  } catch (error) {
    console.error('Error processing contact form:', error);
    return res.status(500).json({ 
      error: 'Failed to process submission',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}
