/**
 * Email Threads API
 * 
 * GET /api/email/threads - List threads with filtering
 * GET /api/email/threads?id=xxx - Get single thread with messages
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Simple auth check (in production, use proper admin auth)
  const authHeader = req.headers.authorization;
  const expectedKey = process.env.CRON_SECRET || process.env.ADMIN_API_KEY;
  
  // Allow unauthenticated access for now (admin dashboard)
  // TODO: Implement proper admin authentication
  
  if (!supabaseUrl || !supabaseKey) {
    return res.status(500).json({ error: 'Database not configured' });
  }

  const supabase = createClient(supabaseUrl, supabaseKey);

  try {
    const { id, status, category, urgency, limit, offset } = req.query;

    // Get single thread with messages
    if (id && typeof id === 'string') {
      const { data: thread, error: threadError } = await supabase
        .from('email_threads')
        .select('*')
        .eq('id', id)
        .single();

      if (threadError || !thread) {
        return res.status(404).json({ error: 'Thread not found' });
      }

      // Get messages
      const { data: messages } = await supabase
        .from('email_messages')
        .select('*')
        .eq('thread_id', id)
        .order('created_at', { ascending: true });

      // Get actions
      const { data: actions } = await supabase
        .from('email_actions')
        .select('*')
        .eq('thread_id', id)
        .order('created_at', { ascending: true });

      return res.status(200).json({
        thread,
        messages: messages || [],
        actions: actions || [],
      });
    }

    // List threads with filtering
    let query = supabase
      .from('email_threads')
      .select('*', { count: 'exact' })
      .order('created_at', { ascending: false });

    // Apply filters
    if (status && typeof status === 'string') {
      if (status === 'pending') {
        query = query.eq('status', 'pending_approval');
      } else if (status === 'needs_action') {
        query = query.in('status', ['pending_approval', 'escalated', 'open']);
      } else {
        query = query.eq('status', status);
      }
    }

    if (category && typeof category === 'string') {
      query = query.eq('category', category);
    }

    if (urgency && typeof urgency === 'string') {
      query = query.eq('urgency', urgency);
    }

    // Pagination
    const limitNum = Math.min(parseInt(limit as string) || 50, 100);
    const offsetNum = parseInt(offset as string) || 0;
    
    query = query.range(offsetNum, offsetNum + limitNum - 1);

    const { data: threads, error, count } = await query;

    if (error) {
      console.error('Failed to fetch threads:', error);
      return res.status(500).json({ error: 'Failed to fetch threads' });
    }

    return res.status(200).json({
      threads: threads || [],
      total: count || 0,
      limit: limitNum,
      offset: offsetNum,
    });

  } catch (error) {
    console.error('Threads API error:', error);
    return res.status(500).json({
      error: 'Failed to fetch threads',
      details: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}
