/**
 * Visual Commons - Stats Endpoint
 * 
 * Returns user contribution statistics and impact metrics.
 * 
 * GET /api/visual/stats
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

// Initialize Supabase
const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '';
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || '';

function getSupabase() {
  return createClient(supabaseUrl, supabaseKey);
}

// Badge definitions
const BADGES = [
  { id: 'first_light', name: 'First Light 💡', requirement: 1, description: 'Generated your first visual' },
  { id: 'visual_pioneer', name: 'Visual Pioneer 🎨', requirement: 10, description: 'Generated 10 visuals' },
  { id: 'illuminator', name: 'Illuminator ✨', requirement: 50, description: 'Generated 50 visuals' },
  { id: 'master_illuminator', name: 'Master Illuminator 🌟', requirement: 100, description: 'Generated 100 visuals' },
  { id: 'helper', name: 'Helper 🤝', requirement: 100, type: 'helped', description: 'Helped 100 learners' },
  { id: 'community_builder', name: 'Community Builder 🏗️', requirement: 1000, type: 'helped', description: 'Helped 1,000 learners' },
  { id: 'legend', name: 'Legend 🏆', requirement: 10000, type: 'helped', description: 'Helped 10,000 learners' }
];

function calculateBadges(contributed: number, helped: number): typeof BADGES {
  return BADGES.filter(badge => {
    if (badge.type === 'helped') {
      return helped >= badge.requirement;
    }
    return contributed >= badge.requirement;
  });
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Only allow GET
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const supabase = getSupabase();
    
    // Get user from auth header (if authenticated)
    const authHeader = req.headers.authorization;
    let userId: string | null = null;
    
    if (authHeader?.startsWith('Bearer ')) {
      const token = authHeader.substring(7);
      const { data: { user } } = await supabase.auth.getUser(token);
      userId = user?.id || null;
    }

    // If no authenticated user, return global stats
    if (!userId) {
      // Get global stats
      const { data: globalStats } = await supabase
        .from('visual_commons')
        .select('id', { count: 'exact' })
        .eq('status', 'active');

      const { data: helpedStats } = await supabase
        .rpc('get_total_learners_helped');

      const { data: topContributors } = await supabase
        .from('user_visual_contributions')
        .select('user_id, total_contributed, total_learners_helped')
        .order('total_contributed', { ascending: false })
        .limit(10);

      return res.status(200).json({
        authenticated: false,
        global: {
          totalVisuals: globalStats?.length || 0,
          totalLearnersHelped: helpedStats || 0,
          topContributors: topContributors || []
        },
        message: 'Sign in to track your personal contributions'
      });
    }

    // Get user's stats
    const { data: userStats } = await supabase
      .from('user_visual_contributions')
      .select('*')
      .eq('user_id', userId)
      .maybeSingle();

    // Get user's recent contributions
    const { data: recentContributions } = await supabase
      .from('visual_commons')
      .select('id, topic, phase, unique_learners_helped, created_at')
      .eq('generated_by', userId)
      .order('created_at', { ascending: false })
      .limit(5);

    // Calculate badges
    const contributed = userStats?.total_contributed || 0;
    const helped = userStats?.total_learners_helped || 0;
    const earnedBadges = calculateBadges(contributed, helped);

    // Get user's rank
    const { data: rankData } = await supabase
      .from('user_visual_contributions')
      .select('user_id')
      .order('total_contributed', { ascending: false });

    const rank = rankData?.findIndex(r => r.user_id === userId) ?? -1;

    return res.status(200).json({
      authenticated: true,
      userId,
      stats: {
        totalContributed: contributed,
        totalLearnersHelped: helped,
        contributionsThisWeek: userStats?.contributions_this_week || 0,
        contributionsThisMonth: userStats?.contributions_this_month || 0
      },
      badges: earnedBadges,
      nextBadge: BADGES.find(b => {
        if (b.type === 'helped') return helped < b.requirement;
        return contributed < b.requirement;
      }),
      rank: rank >= 0 ? rank + 1 : null,
      recentContributions: recentContributions || [],
      impact: {
        message: helped > 0 
          ? `Your visuals have helped ${helped.toLocaleString()} learners understand complex topics!`
          : 'Generate your first visual to start helping others learn!'
      }
    });

  } catch (error: any) {
    console.error('Stats error:', error);
    return res.status(500).json({ 
      error: 'Failed to fetch stats',
      message: error.message 
    });
  }
}
