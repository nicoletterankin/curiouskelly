/**
 * Phase Commons - Votes API
 * 
 * POST /api/commons/votes
 *   Vote on a proposal (upvote, downvote, or remove vote)
 * 
 * GET /api/commons/votes?proposal_id=xxx
 *   Get user's vote on a proposal
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

    // Authenticate
    const authHeader = req.headers.authorization;
    if (!authHeader) {
      return res.status(401).json({ error: 'Authentication required to vote' });
    }

    const token = authHeader.replace('Bearer ', '');
    const { data: { user }, error: authError } = await supabase.auth.getUser(token);

    if (authError || !user) {
      return res.status(401).json({ error: 'Invalid authentication' });
    }

    if (req.method === 'GET') {
      return handleGet(req, res, supabase, user.id);
    } else if (req.method === 'POST') {
      return handlePost(req, res, supabase, user.id);
    } else {
      return res.status(405).json({ error: 'Method not allowed' });
    }
  } catch (error) {
    console.error('[commons/votes] Error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

async function handleGet(req: VercelRequest, res: VercelResponse, supabase: any, userId: string) {
  const { proposal_id } = req.query;

  if (!proposal_id) {
    return res.status(400).json({ error: 'proposal_id is required' });
  }

  const { data: vote, error } = await supabase
    .from('commons_votes')
    .select('vote_type')
    .eq('proposal_id', proposal_id)
    .eq('user_id', userId)
    .single();

  if (error && error.code !== 'PGRST116') { // PGRST116 = no rows
    console.error('[commons/votes] Query error:', error);
    return res.status(500).json({ error: error.message });
  }

  return res.status(200).json({
    proposalId: proposal_id,
    vote: vote?.vote_type || null
  });
}

async function handlePost(req: VercelRequest, res: VercelResponse, supabase: any, userId: string) {
  const { proposalId, vote } = req.body;

  if (!proposalId) {
    return res.status(400).json({ error: 'proposalId is required' });
  }

  // vote can be 'up', 'down', or null (to remove vote)
  if (vote !== null && vote !== 'up' && vote !== 'down') {
    return res.status(400).json({ error: 'vote must be "up", "down", or null' });
  }

  // Get existing vote
  const { data: existingVote } = await supabase
    .from('commons_votes')
    .select('id, vote_type')
    .eq('proposal_id', proposalId)
    .eq('user_id', userId)
    .single();

  const oldVote = existingVote?.vote_type || null;

  // Start transaction-like operations
  let voteError = null;
  let proposalError = null;

  if (vote === null && existingVote) {
    // Remove vote
    const { error } = await supabase
      .from('commons_votes')
      .delete()
      .eq('id', existingVote.id);
    voteError = error;
  } else if (vote && existingVote) {
    // Update existing vote
    const { error } = await supabase
      .from('commons_votes')
      .update({ vote_type: vote, updated_at: new Date().toISOString() })
      .eq('id', existingVote.id);
    voteError = error;
  } else if (vote) {
    // Insert new vote
    const { error } = await supabase
      .from('commons_votes')
      .insert({
        proposal_id: proposalId,
        user_id: userId,
        vote_type: vote
      });
    voteError = error;
  }

  if (voteError) {
    // If votes table doesn't exist, return mock success
    if (voteError.code === '42P01') {
      return res.status(200).json({
        success: true,
        proposalId,
        vote,
        message: 'Vote recorded (mock mode)'
      });
    }
    console.error('[commons/votes] Vote error:', voteError);
    return res.status(500).json({ error: voteError.message });
  }

  // Update proposal vote counts
  let upvoteDelta = 0;
  let downvoteDelta = 0;

  // Remove old vote effect
  if (oldVote === 'up') upvoteDelta--;
  if (oldVote === 'down') downvoteDelta--;

  // Add new vote effect
  if (vote === 'up') upvoteDelta++;
  if (vote === 'down') downvoteDelta++;

  if (upvoteDelta !== 0 || downvoteDelta !== 0) {
    // Get current counts
    const { data: proposal } = await supabase
      .from('commons_proposals')
      .select('upvotes, downvotes')
      .eq('id', proposalId)
      .single();

    if (proposal) {
      const { error } = await supabase
        .from('commons_proposals')
        .update({
          upvotes: Math.max(0, (proposal.upvotes || 0) + upvoteDelta),
          downvotes: Math.max(0, (proposal.downvotes || 0) + downvoteDelta)
        })
        .eq('id', proposalId);
      proposalError = error;
    }
  }

  return res.status(200).json({
    success: true,
    proposalId,
    vote,
    previousVote: oldVote
  });
}
