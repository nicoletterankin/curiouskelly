/**
 * Daily Email Digest Cron
 * 
 * Sends a summary of pending emails to nicoletterankin@gmail.com
 * 
 * Recommended schedule: 8am EST daily
 * Vercel cron: "0 13 * * *" (1pm UTC = 8am EST)
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';
import { sendDailyDigest } from '../../lib/email/escalation';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Verify cron secret
  const authHeader = req.headers.authorization;
  const cronSecret = process.env.CRON_SECRET;
  
  if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  if (!supabaseUrl || !supabaseKey) {
    return res.status(500).json({ error: 'Database not configured' });
  }

  const supabase = createClient(supabaseUrl, supabaseKey);

  try {
    // Get all threads that need attention
    const { data: pendingThreads, error } = await supabase
      .from('email_threads')
      .select('id, from_email, subject, category, urgency, created_at, ai_summary')
      .in('status', ['pending_approval', 'escalated', 'open'])
      .order('urgency', { ascending: false })
      .order('created_at', { ascending: true });

    if (error) {
      console.error('Failed to fetch pending threads:', error);
      return res.status(500).json({ error: 'Failed to fetch threads' });
    }

    if (!pendingThreads || pendingThreads.length === 0) {
      console.log('[email-digest] No pending items');
      return res.status(200).json({ 
        success: true, 
        message: 'No pending items',
        count: 0 
      });
    }

    // Format for digest
    const digestItems = pendingThreads.map(thread => ({
      threadId: thread.id,
      from: thread.from_email,
      subject: thread.subject,
      category: thread.category,
      urgency: thread.urgency,
      receivedAt: new Date(thread.created_at),
      aiSummary: thread.ai_summary || '',
    }));

    // Send the digest
    const result = await sendDailyDigest(digestItems);

    if (result.success) {
      console.log(`[email-digest] Sent digest with ${digestItems.length} items`);
      return res.status(200).json({
        success: true,
        message: `Digest sent with ${digestItems.length} items`,
        count: digestItems.length,
        messageId: result.messageId,
      });
    } else {
      console.error('[email-digest] Failed to send:', result.error);
      return res.status(500).json({
        error: 'Failed to send digest',
        details: result.error,
      });
    }

  } catch (error) {
    console.error('[email-digest] Error:', error);
    return res.status(500).json({
      error: 'Digest cron failed',
      details: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}
