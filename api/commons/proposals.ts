/**
 * Phase Commons - Proposals API
 * 
 * GET /api/commons/proposals?address=017.hook.talk
 *   Returns proposals for a specific content address
 * 
 * POST /api/commons/proposals
 *   Submit a new proposal
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY || '';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  try {
    const supabase = createClient(supabaseUrl, supabaseKey);

    if (req.method === 'GET') {
      return handleGet(req, res, supabase);
    } else if (req.method === 'POST') {
      return handlePost(req, res, supabase);
    } else {
      return res.status(405).json({ error: 'Method not allowed' });
    }
  } catch (error) {
    console.error('[commons/proposals] Error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

async function handleGet(req: VercelRequest, res: VercelResponse, supabase: any) {
  const { address, day, phase, status = 'open' } = req.query;

  // Build query
  let query = supabase
    .from('commons_proposals')
    .select(`
      id,
      title,
      description,
      type,
      status,
      target_atoms,
      proposed_changes,
      requires_audio_regen,
      requires_video_regen,
      upvotes,
      downvotes,
      created_at,
      user_id,
      users:user_id (
        id,
        raw_user_meta_data
      )
    `)
    .eq('status', status)
    .order('upvotes', { ascending: false });

  // Filter by content address
  if (address) {
    query = query.contains('target_atoms', [address]);
  }

  // Or filter by day/phase
  if (day) {
    const paddedDay = String(day).padStart(3, '0');
    query = query.filter('target_atoms', 'cs', `{${paddedDay}.`);
  }

  const { data: proposals, error } = await query.limit(20);

  if (error) {
    console.error('[commons/proposals] Query error:', error);
    
    // Fallback to mock data if table doesn't exist yet
    if (error.code === '42P01') {
      return res.status(200).json({
        proposals: getMockProposals(address as string),
        source: 'mock'
      });
    }
    
    return res.status(500).json({ error: error.message });
  }

  // Format response
  const formattedProposals = (proposals || []).map((p: any) => ({
    id: p.id,
    title: p.title,
    description: p.description,
    type: p.type,
    status: p.status,
    targetAtoms: p.target_atoms,
    proposedChanges: p.proposed_changes,
    requiresAudioRegen: p.requires_audio_regen,
    upvotes: p.upvotes || 0,
    downvotes: p.downvotes || 0,
    createdAt: p.created_at,
    author: p.users?.raw_user_meta_data?.full_name || 
            p.users?.raw_user_meta_data?.name || 
            'Anonymous'
  }));

  return res.status(200).json({
    proposals: formattedProposals,
    count: formattedProposals.length,
    source: 'database'
  });
}

async function handlePost(req: VercelRequest, res: VercelResponse, supabase: any) {
  // Get user from auth header
  const authHeader = req.headers.authorization;
  if (!authHeader) {
    return res.status(401).json({ error: 'Authentication required' });
  }

  const token = authHeader.replace('Bearer ', '');
  const { data: { user }, error: authError } = await supabase.auth.getUser(token);

  if (authError || !user) {
    return res.status(401).json({ error: 'Invalid authentication' });
  }

  const { targetAtoms, proposedChanges, type, title, rationale } = req.body;

  if (!targetAtoms?.length || !proposedChanges || !type || !title) {
    return res.status(400).json({ 
      error: 'Missing required fields: targetAtoms, proposedChanges, type, title' 
    });
  }

  // Validate proposal type
  const validTypes = ['enhance', 'correct', 'simplify', 'expand', 'typo'];
  if (!validTypes.includes(type)) {
    return res.status(400).json({ error: `Invalid type. Must be one of: ${validTypes.join(', ')}` });
  }

  // Insert proposal
  const { data: proposal, error } = await supabase
    .from('commons_proposals')
    .insert({
      user_id: user.id,
      title,
      description: rationale || '',
      type,
      status: 'open',
      target_atoms: targetAtoms,
      proposed_changes: proposedChanges,
      requires_audio_regen: true, // Text changes always need audio regen
      requires_video_regen: false,
      upvotes: 0,
      downvotes: 0
    })
    .select()
    .single();

  if (error) {
    console.error('[commons/proposals] Insert error:', error);
    
    // If table doesn't exist, return mock success
    if (error.code === '42P01') {
      return res.status(200).json({
        success: true,
        proposal: { id: 'mock-' + Date.now(), ...req.body },
        message: 'Proposal submitted (mock mode - table not yet created)'
      });
    }
    
    return res.status(500).json({ error: error.message });
  }

  return res.status(201).json({
    success: true,
    proposal,
    message: 'Proposal submitted successfully'
  });
}

// Mock data for when database isn't set up yet
function getMockProposals(address?: string) {
  const mockProposals = [
    {
      id: 'mock-1',
      title: 'Add clarification about timing',
      description: 'Makes the concept clearer for younger learners',
      type: 'enhance',
      status: 'open',
      targetAtoms: [address || '017.hook.talk'],
      upvotes: 15,
      downvotes: 2,
      createdAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
      author: 'curious_learner'
    },
    {
      id: 'mock-2',
      title: 'Simplify technical term',
      description: 'Use everyday language instead of jargon',
      type: 'simplify',
      status: 'open',
      targetAtoms: [address || '017.hook.talk'],
      upvotes: 8,
      downvotes: 1,
      createdAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
      author: 'educator_mom'
    }
  ];
  
  return mockProposals;
}
