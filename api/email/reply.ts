/**
 * Manual Reply to Email Thread
 * 
 * POST /api/email/reply
 * 
 * Body:
 * - threadId: UUID of the thread to reply to
 * - subject: Email subject line
 * - body: Email body text
 * - closeThread: Whether to close the thread after sending (default: true)
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const RESEND_API_URL = 'https://api.resend.com/emails';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

interface ReplyRequest {
  threadId: string;
  subject: string;
  body: string;
  closeThread?: boolean;
  sender?: string;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  if (!supabaseUrl || !supabaseKey) {
    return res.status(500).json({ error: 'Database not configured' });
  }

  const resendApiKey = process.env.RESEND_API_KEY;
  if (!resendApiKey) {
    return res.status(500).json({ error: 'Email service not configured' });
  }

  const supabase = createClient(supabaseUrl, supabaseKey);
  const body = req.body as ReplyRequest;

  if (!body.threadId || !body.subject || !body.body) {
    return res.status(400).json({ 
      error: 'Missing required fields',
      required: ['threadId', 'subject', 'body']
    });
  }

  try {
    // Get the thread
    const { data: thread, error: threadError } = await supabase
      .from('email_threads')
      .select('*')
      .eq('id', body.threadId)
      .single();

    if (threadError || !thread) {
      return res.status(404).json({ error: 'Thread not found' });
    }

    const htmlBody = textToHtml(body.body);

    // Send the email
    const sendResponse = await fetch(RESEND_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${resendApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'Kelly <hello@curiouskelly.com>',
        to: thread.from_email,
        subject: body.subject,
        text: body.body,
        html: htmlBody,
        reply_to: 'hello@curiouskelly.com',
      }),
    });

    const sendData = await sendResponse.json();

    if (!sendResponse.ok) {
      console.error('Failed to send reply:', sendData);
      return res.status(500).json({ 
        error: 'Failed to send email',
        details: sendData.message || 'Unknown error'
      });
    }

    // Store the outbound message
    const { data: message } = await supabase
      .from('email_messages')
      .insert({
        thread_id: body.threadId,
        direction: 'outbound',
        from_email: 'hello@curiouskelly.com',
        to_email: thread.from_email,
        subject: body.subject,
        body_text: body.body,
        body_html: htmlBody,
        resend_message_id: sendData.id,
        resend_status: 'sent',
        sent_at: new Date().toISOString(),
        is_approved: true,
        approved_by: body.sender || 'admin',
        approved_at: new Date().toISOString(),
      })
      .select()
      .single();

    // Update thread status
    const newStatus = body.closeThread !== false ? 'closed' : 'responded';
    
    await supabase
      .from('email_threads')
      .update({
        status: newStatus,
        responded_at: new Date().toISOString(),
      })
      .eq('id', body.threadId);

    // Log the action
    await supabase.from('email_actions').insert({
      thread_id: body.threadId,
      message_id: message?.id,
      action: 'sent',
      actor: body.sender || 'admin',
      details: {
        manual: true,
        resend_id: sendData.id,
      },
    });

    if (body.closeThread !== false) {
      await supabase.from('email_actions').insert({
        thread_id: body.threadId,
        action: 'closed',
        actor: body.sender || 'admin',
        details: {},
      });
    }

    console.log(`[reply] Sent manual reply to ${thread.from_email} for thread ${body.threadId}`);

    return res.status(200).json({
      success: true,
      messageId: sendData.id,
      to: thread.from_email,
      status: newStatus,
    });

  } catch (error) {
    console.error('Reply API error:', error);
    return res.status(500).json({
      error: 'Failed to send reply',
      details: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}

function textToHtml(text: string): string {
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>');

  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: Georgia, 'Times New Roman', serif; background-color: #f9fafb;">
  <div style="max-width: 520px; margin: 0 auto; padding: 40px 20px;">
    <div style="background: white; border-radius: 16px; padding: 32px; box-shadow: 0 4px 16px rgba(0,0,0,0.08);">
      <div style="font-size: 18px; color: #374151; line-height: 1.8;">
        <p>${html}</p>
      </div>
    </div>
    <div style="text-align: center; margin-top: 32px; padding-top: 24px; border-top: 1px solid #e5e7eb;">
      <p style="font-size: 13px; color: #9ca3af; margin: 0;">
        ✨ Curious Kelly • <a href="https://curiouskelly.com" style="color: #d4a574;">curiouskelly.com</a>
      </p>
    </div>
  </div>
</body>
</html>`.trim();
}
