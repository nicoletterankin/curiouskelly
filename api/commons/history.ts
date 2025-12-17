/**
 * Phase Commons - Content History API
 * 
 * GET /api/commons/history?address=017.hook.talk
 *   Returns version history for a specific content address
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY || '';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const supabase = createClient(supabaseUrl, supabaseKey);
    const { address } = req.query;

    if (!address) {
      return res.status(400).json({ error: 'address is required' });
    }

    // First get the current content atom
    const { data: atom, error: atomError } = await supabase
      .from('content_atoms')
      .select('id, version, text_content, metadata, change_source, change_reason, updated_at')
      .eq('content_address', address)
      .eq('is_live', true)
      .single();

    if (atomError) {
      console.error('[commons/history] Atom query error:', atomError);
      
      // Fallback to mock data if table doesn't exist yet
      if (atomError.code === '42P01' || atomError.code === 'PGRST116') {
        return res.status(200).json({
          history: getMockHistory(address as string),
          source: 'mock'
        });
      }
      
      return res.status(500).json({ error: atomError.message });
    }

    // Get history for this atom
    const { data: history, error: historyError } = await supabase
      .from('content_history')
      .select(`
        id,
        version,
        text_content,
        metadata,
        change_source,
        change_reason,
        change_reference,
        created_at
      `)
      .eq('content_atom_id', atom.id)
      .order('version', { ascending: false });

    if (historyError) {
      console.error('[commons/history] History query error:', historyError);
    }

    // Combine current version with history
    const allVersions = [
      {
        version: atom.version,
        content: atom.text_content,
        metadata: atom.metadata,
        source: atom.change_source,
        reason: atom.change_reason,
        createdAt: atom.updated_at,
        isCurrent: true
      },
      ...(history || []).map((h: any) => ({
        version: h.version,
        content: h.text_content,
        metadata: h.metadata,
        source: h.change_source,
        reason: h.change_reason,
        proposalId: h.change_reference,
        createdAt: h.created_at,
        isCurrent: false
      }))
    ];

    return res.status(200).json({
      address,
      history: allVersions,
      currentVersion: atom.version,
      source: 'database'
    });

  } catch (error) {
    console.error('[commons/history] Error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

// Mock data for when database isn't set up yet
function getMockHistory(address: string) {
  const now = new Date();
  return [
    {
      version: 3,
      content: 'Current version of the content with recent improvements.',
      source: 'commons_proposal',
      reason: 'Added clarification per community proposal',
      createdAt: new Date(now.getTime() - 3 * 24 * 60 * 60 * 1000).toISOString(),
      isCurrent: true
    },
    {
      version: 2,
      content: 'Previous version with minor changes.',
      source: 'commons_proposal',
      reason: 'Simplified language',
      createdAt: new Date(now.getTime() - 5 * 24 * 60 * 60 * 1000).toISOString(),
      isCurrent: false
    },
    {
      version: 1,
      content: 'Original launch content.',
      source: 'initial_seed',
      reason: null,
      createdAt: new Date('2025-12-17').toISOString(),
      isCurrent: false
    }
  ];
}
