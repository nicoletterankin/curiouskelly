/**
 * Visual Commons - Check Endpoint
 * 
 * Checks if a visual exists for the given context.
 * Returns cached visual URL or generation options.
 * 
 * GET /api/visual/check?day=17&phase=hook&age=6-12&type=infographic
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';
import crypto from 'crypto';

// Types
interface VisualContext {
  dayNumber: number;
  phase: string;
  ageGroup: string;
  visualType: string;
  style?: string;
}

interface CachedVisual {
  id: string;
  publicUrl: string;
  thumbnailUrl: string | null;
  generatedBy: {
    displayName: string;
    isAnonymous: boolean;
  };
  helpedCount: number;
  createdAt: string;
}

// Initialize Supabase
const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '';
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || '';

function getSupabase() {
  return createClient(supabaseUrl, supabaseKey);
}

// Generate content hash (must match client-side implementation)
function generateVisualHash(context: VisualContext): string {
  const normalized = {
    d: context.dayNumber,
    p: context.phase.toLowerCase(),
    a: context.ageGroup || 'all',
    t: context.visualType || 'infographic',
    s: context.style || 'default',
    ver: '1'
  };
  
  const canonical = JSON.stringify(normalized, Object.keys(normalized).sort());
  return crypto.createHash('sha256').update(canonical).digest('hex');
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Only allow GET
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Parse query parameters
    const dayNumber = parseInt(req.query.day as string);
    const phase = req.query.phase as string;
    const ageGroup = (req.query.age as string) || 'all';
    const visualType = (req.query.type as string) || 'infographic';
    const style = (req.query.style as string) || 'default';

    // Validate required params
    if (!dayNumber || !phase) {
      return res.status(400).json({ 
        error: 'Missing required parameters',
        required: ['day', 'phase'],
        optional: ['age', 'type', 'style']
      });
    }

    if (dayNumber < 1 || dayNumber > 365) {
      return res.status(400).json({ error: 'Day must be between 1 and 365' });
    }

    const validPhases = ['hook', 'cliff', 'fact1', 'fact2', 'fact3', 'wisdom', 'outro', 'complete'];
    if (!validPhases.includes(phase.toLowerCase())) {
      return res.status(400).json({ error: `Invalid phase. Must be one of: ${validPhases.join(', ')}` });
    }

    // Build context and generate hash
    const context: VisualContext = {
      dayNumber,
      phase: phase.toLowerCase(),
      ageGroup,
      visualType,
      style
    };

    const contentHash = generateVisualHash(context);

    // Check cache
    const supabase = getSupabase();
    const { data: visual, error } = await supabase
      .from('visual_commons')
      .select(`
        id,
        public_url,
        thumbnail_url,
        generated_by_display_name,
        generated_by,
        unique_learners_helped,
        created_at
      `)
      .eq('content_hash', contentHash)
      .eq('status', 'active')
      .maybeSingle();

    if (error) {
      console.error('Supabase error:', error);
      return res.status(500).json({ error: 'Database error', details: error.message });
    }

    // Cache hit
    if (visual) {
      const cachedVisual: CachedVisual = {
        id: visual.id,
        publicUrl: visual.public_url,
        thumbnailUrl: visual.thumbnail_url,
        generatedBy: {
          displayName: visual.generated_by_display_name || 'A curious learner',
          isAnonymous: !visual.generated_by
        },
        helpedCount: visual.unique_learners_helped || 0,
        createdAt: visual.created_at
      };

      return res.status(200).json({
        exists: true,
        contentHash,
        visual: cachedVisual
      });
    }

    // Cache miss - check if generation is possible
    // For now, assume generation is always possible
    // In production, check rate limits, API key availability, etc.
    
    return res.status(200).json({
      exists: false,
      contentHash,
      canGenerate: true,
      estimatedCost: 0, // Gemini infographics are free
      keySource: 'platform', // Will be updated based on user's BYOK status
      context: {
        dayNumber,
        phase,
        ageGroup,
        visualType
      }
    });

  } catch (error: any) {
    console.error('Visual check error:', error);
    return res.status(500).json({ 
      error: 'Internal server error',
      message: error.message 
    });
  }
}
