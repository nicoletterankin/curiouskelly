import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

/**
 * Log a community contribution (BYOK usage that powered Kelly)
 * 
 * POST /api/contribution
 * {
 *   provider: 'openai' | 'heygen' | 'elevenlabs' | ...,
 *   resource_type: 'chat' | 'video' | 'voice' | 'image',
 *   day_number?: number,
 *   phase?: string,
 *   estimated_cost_cents?: number,
 *   credits_used?: number
 * }
 * 
 * GET /api/contribution
 * Returns community contribution summary
 */
export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  try {
    if (req.method === 'GET') {
      // Return contribution summary
      const { data, error } = await supabase
        .from('v_contribution_summary')
        .select('*')
        .single();

      if (error) {
        // If view doesn't exist yet or is empty, return zeros
        return res.status(200).json({
          total_contributions: 0,
          contributors: 0,
          value_dollars: 0,
          lessons_touched: 0
        });
      }

      return res.status(200).json(data);
    }

    if (req.method === 'POST') {
      const {
        provider,
        resource_type,
        day_number,
        phase,
        estimated_cost_cents = 0,
        credits_used = 1,
        contributor_id = null,
        is_anonymous = true
      } = req.body;

      if (!provider || !resource_type) {
        return res.status(400).json({
          error: 'Missing required fields: provider, resource_type'
        });
      }

      const { data, error } = await supabase
        .from('community_contributions')
        .insert({
          provider,
          resource_type,
          day_number,
          phase,
          estimated_cost_cents,
          credits_used,
          contributor_id,
          is_anonymous
        })
        .select()
        .single();

      if (error) {
        console.error('Contribution logging error:', error);
        return res.status(500).json({ error: 'Failed to log contribution' });
      }

      return res.status(201).json({
        success: true,
        contribution_id: data.id,
        message: 'Thank you for powering Kelly! 🎁'
      });
    }

    return res.status(405).json({ error: 'Method not allowed' });

  } catch (err: any) {
    console.error('Contribution API error:', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
