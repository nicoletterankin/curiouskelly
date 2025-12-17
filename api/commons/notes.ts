/**
 * Phase Commons - Community Notes API
 * 
 * GET /api/commons/notes?address=017.hook.talk
 *   Returns community notes for a specific content address
 * 
 * POST /api/commons/notes
 *   Add a new community note
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
    console.error('[commons/notes] Error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

async function handleGet(req: VercelRequest, res: VercelResponse, supabase: any) {
  const { address, day, phase } = req.query;

  // Build query
  let query = supabase
    .from('commons_notes')
    .select(`
      id,
      content_address,
      note_type,
      content,
      sources,
      is_verified,
      is_featured,
      helpful_count,
      insightful_count,
      created_at,
      user_id,
      users:user_id (
        id,
        raw_user_meta_data
      )
    `)
    .order('is_featured', { ascending: false })
    .order('helpful_count', { ascending: false });

  // Filter by content address
  if (address) {
    query = query.eq('content_address', address);
  } else if (day && phase) {
    const paddedDay = String(day).padStart(3, '0');
    query = query.like('content_address', `${paddedDay}.${phase}%`);
  }

  const { data: notes, error } = await query.limit(20);

  if (error) {
    console.error('[commons/notes] Query error:', error);
    
    // Fallback to mock data if table doesn't exist yet
    if (error.code === '42P01') {
      return res.status(200).json({
        notes: getMockNotes(address as string),
        source: 'mock'
      });
    }
    
    return res.status(500).json({ error: error.message });
  }

  // Format response
  const formattedNotes = (notes || []).map((n: any) => ({
    id: n.id,
    contentAddress: n.content_address,
    type: n.note_type,
    content: n.content,
    sources: n.sources || [],
    isVerified: n.is_verified,
    isFeatured: n.is_featured,
    reactions: {
      helpful: n.helpful_count || 0,
      insightful: n.insightful_count || 0
    },
    createdAt: n.created_at,
    author: n.users?.raw_user_meta_data?.full_name || 
            n.users?.raw_user_meta_data?.name || 
            'Anonymous'
  }));

  return res.status(200).json({
    notes: formattedNotes,
    count: formattedNotes.length,
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

  const { contentAddress, noteType, content, sources } = req.body;

  if (!contentAddress || !noteType || !content) {
    return res.status(400).json({ 
      error: 'Missing required fields: contentAddress, noteType, content' 
    });
  }

  // Validate note type
  const validTypes = [
    'expert_context',
    'historical_note',
    'source_citation',
    'teaching_tip',
    'common_misconception',
    'real_world_example'
  ];
  
  if (!validTypes.includes(noteType)) {
    return res.status(400).json({ error: `Invalid noteType. Must be one of: ${validTypes.join(', ')}` });
  }

  // Insert note
  const { data: note, error } = await supabase
    .from('commons_notes')
    .insert({
      user_id: user.id,
      content_address: contentAddress,
      note_type: noteType,
      content,
      sources: sources || [],
      is_verified: false,
      is_featured: false,
      helpful_count: 0,
      insightful_count: 0
    })
    .select()
    .single();

  if (error) {
    console.error('[commons/notes] Insert error:', error);
    
    // If table doesn't exist, return mock success
    if (error.code === '42P01') {
      return res.status(200).json({
        success: true,
        note: { id: 'mock-' + Date.now(), ...req.body },
        message: 'Note submitted (mock mode - table not yet created)'
      });
    }
    
    return res.status(500).json({ error: error.message });
  }

  return res.status(201).json({
    success: true,
    note,
    message: 'Note added successfully'
  });
}

// Mock data for when database isn't set up yet
function getMockNotes(address?: string) {
  return [
    {
      id: 'mock-n1',
      contentAddress: address || '017.hook.talk',
      type: 'expert_context',
      content: 'The research on this topic is robust. Key studies have shown significant effects that support what Kelly teaches here.',
      sources: ['https://nature.com/articles/example'],
      isVerified: true,
      isFeatured: false,
      reactions: { helpful: 12, insightful: 8 },
      createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
      author: 'science_prof'
    },
    {
      id: 'mock-n2',
      contentAddress: address || '017.hook.talk',
      type: 'teaching_tip',
      content: 'For younger learners, using a simple analogy works really well here. They grasp the concept faster when framed as something familiar.',
      sources: [],
      isVerified: false,
      isFeatured: false,
      reactions: { helpful: 7, insightful: 3 },
      createdAt: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
      author: 'educator_mom'
    }
  ];
}
