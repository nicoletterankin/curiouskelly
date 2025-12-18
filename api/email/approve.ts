/**
 * Approve and Send Draft Email
 * 
 * POST /api/email/approve
 * 
 * Body:
 * - threadId: UUID of the thread
 * - messageId: UUID of the draft message (optional, uses latest draft if not provided)
 * - editedSubject: Optional edited subject line
 * - editedBody: Optional edited body text
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const RESEND_API_URL = 'https://api.resend.com/emails';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

interface ApproveRequest {
  threadId: string;
  messageId?: string;
  editedSubject?: string;
  editedBody?: string;
  approver?: string;
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
  const body = req.body as ApproveRequest;

  if (!body.threadId) {
    return res.status(400).json({ error: 'threadId is required' });
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

    // Get the draft message
    let draftQuery = supabase
      .from('email_messages')
      .select('*')
      .eq('thread_id', body.threadId)
      .eq('direction', 'draft')
      .eq('is_approved', false);

    if (body.messageId) {
      draftQuery = draftQuery.eq('id', body.messageId);
    }

    const { data: drafts, error: draftError } = await draftQuery
      .order('created_at', { ascending: false })
      .limit(1);

    if (draftError || !drafts || drafts.length === 0) {
      return res.status(404).json({ error: 'No draft found for this thread' });
    }

    const draft = drafts[0];

    // Use edited content or original
    const subject = body.editedSubject || draft.subject;
    const bodyText = body.editedBody || draft.body_text;
    const bodyHtml = body.editedBody 
      ? textToHtml(body.editedBody) 
      : draft.body_html;

    // Send the email
    const sendResponse = await fetch(RESEND_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${resendApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: 'Kelly <hello@curiouskelly.com>',
        to: draft.to_email,
        subject,
        text: bodyText,
        html: bodyHtml,
        reply_to: 'hello@curiouskelly.com',
      }),
    });

    const sendData = await sendResponse.json();

    if (!sendResponse.ok) {
      console.error('Failed to send email:', sendData);
      return res.status(500).json({ 
        error: 'Failed to send email',
        details: sendData.message || 'Unknown error'
      });
    }

    // Update the draft message
    await supabase
      .from('email_messages')
      .update({
        is_approved: true,
        approved_by: body.approver || 'admin',
        approved_at: new Date().toISOString(),
        subject,
        body_text: bodyText,
        body_html: bodyHtml,
        resend_message_id: sendData.id,
        resend_status: 'sent',
        sent_at: new Date().toISOString(),
        direction: 'outbound', // Change from draft to outbound
      })
      .eq('id', draft.id);

    // Update the thread
    await supabase
      .from('email_threads')
      .update({
        status: 'responded',
        responded_at: new Date().toISOString(),
      })
      .eq('id', body.threadId);

    // Log the action
    await supabase.from('email_actions').insert({
      thread_id: body.threadId,
      message_id: draft.id,
      action: 'approved',
      actor: body.approver || 'admin',
      details: {
        edited: !!(body.editedSubject || body.editedBody),
      },
    });

    await supabase.from('email_actions').insert({
      thread_id: body.threadId,
      message_id: draft.id,
      action: 'sent',
      actor: 'system',
      details: {
        resend_id: sendData.id,
      },
    });

    console.log(`[approve] Sent response to ${draft.to_email} for thread ${body.threadId}`);

    return res.status(200).json({
      success: true,
      messageId: sendData.id,
      to: draft.to_email,
      subject,
    });

  } catch (error) {
    console.error('Approve API error:', error);
    return res.status(500).json({
      error: 'Failed to approve and send',
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
