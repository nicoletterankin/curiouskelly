/**
 * Submission Viewer API
 * 
 * Returns submission details for the public response page.
 * Allows the submitter to view what they sent and add follow-up comments.
 * 
 * GET /api/submission/:id - Get submission details
 * POST /api/submission/:id - Add a follow-up comment/file
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const { id } = req.query;

  if (!id || typeof id !== 'string') {
    return res.status(400).json({ error: 'Submission ID required' });
  }

  // Validate UUID format
  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  if (!uuidRegex.test(id)) {
    return res.status(400).json({ error: 'Invalid submission ID' });
  }

  if (!supabaseUrl || !supabaseServiceKey) {
    return res.status(500).json({ error: 'Service not configured' });
  }

  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  if (req.method === 'GET') {
    // Fetch submission
    const { data, error } = await supabase
      .from('contact_submissions')
      .select('id, name, email, subject, message, source, attachments, status, created_at, notes')
      .eq('id', id)
      .single();

    if (error || !data) {
      return res.status(404).json({ error: 'Submission not found' });
    }

    // Return sanitized data (hide internal fields)
    return res.status(200).json({
      id: data.id,
      name: data.name,
      // Mask email for privacy: show first 3 chars + domain
      email: maskEmail(data.email),
      subject: data.subject,
      message: data.message,
      source: data.source,
      attachments: data.attachments || [],
      status: data.status,
      created_at: data.created_at,
      // Only show notes if status is 'replied' (response from Kelly)
      response: data.status === 'replied' ? data.notes : null
    });
  }

  if (req.method === 'POST') {
    // Add follow-up comment
    const { comment, attachments } = req.body;

    if (!comment && (!attachments || attachments.length === 0)) {
      return res.status(400).json({ error: 'Comment or attachments required' });
    }

    // Get current submission
    const { data: submission, error: fetchError } = await supabase
      .from('contact_submissions')
      .select('*')
      .eq('id', id)
      .single();

    if (fetchError || !submission) {
      return res.status(404).json({ error: 'Submission not found' });
    }

    // Append follow-up to notes
    const timestamp = new Date().toISOString();
    const followUp = `\n\n--- Follow-up from submitter (${timestamp}) ---\n${comment || '(Added files)'}`;
    const updatedNotes = (submission.notes || '') + followUp;

    // Handle new attachments if any
    let updatedAttachments = submission.attachments || [];
    if (attachments && attachments.length > 0) {
      for (const att of attachments) {
        try {
          const buffer = Buffer.from(att.content, 'base64');
          const timestamp = Date.now();
          const safeName = att.filename.replace(/[^a-zA-Z0-9.-]/g, '_');
          const storagePath = `${id}/followup_${timestamp}_${safeName}`;

          const { error: uploadError } = await supabase.storage
            .from('submissions')
            .upload(storagePath, buffer, {
              contentType: att.contentType || 'application/octet-stream',
              upsert: false
            });

          if (!uploadError) {
            const { data: urlData } = await supabase.storage
              .from('submissions')
              .createSignedUrl(storagePath, 60 * 60 * 24 * 365);

            updatedAttachments.push({
              filename: att.filename,
              storage_path: storagePath,
              public_url: urlData?.signedUrl || '',
              size_bytes: buffer.length,
              mime_type: att.contentType || 'application/octet-stream',
              is_followup: true
            });
          }
        } catch (e) {
          console.error('Failed to upload follow-up attachment:', e);
        }
      }
    }

    // Update submission
    const { error: updateError } = await supabase
      .from('contact_submissions')
      .update({
        notes: updatedNotes,
        attachments: updatedAttachments,
        updated_at: timestamp
      })
      .eq('id', id);

    if (updateError) {
      return res.status(500).json({ error: 'Failed to save follow-up' });
    }

    return res.status(200).json({ 
      success: true, 
      message: 'Follow-up added successfully' 
    });
  }

  return res.status(405).json({ error: 'Method not allowed' });
}

function maskEmail(email: string): string {
  const [local, domain] = email.split('@');
  if (!domain) return '***';
  const maskedLocal = local.slice(0, 3) + '***';
  return `${maskedLocal}@${domain}`;
}



