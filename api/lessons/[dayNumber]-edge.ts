/**
 * Edge-Optimized Lesson API
 * Uses Edge Config for instant metadata reads (<5ms globally)
 * Falls back to Supabase if Edge Config miss
 * 
 * Performance: <50ms response time globally
 */

import { get } from '@vercel/edge-config';
import { createClient } from '@supabase/supabase-js';
import type { VercelRequest, VercelResponse } from '@vercel/node';

export const config = {
  runtime: 'edge',
};

interface LessonMetadata {
  day: number;
  topic: string;
  emoji: string;
  category: string;
  headline: string;
  universal_truth?: string;
  hasLearn: boolean;
  hasGrow: boolean;
  phases: string[];
  archetypes: string[];
}

export default async function handler(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const pathParts = url.pathname.split('/');
  const dayNumber = pathParts[pathParts.length - 1];
  const day = parseInt(dayNumber);
  
  if (!day || day < 1 || day > 365) {
    return new Response(
      JSON.stringify({ error: 'Invalid day number (1-365)' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } }
    );
  }
  
  const searchParams = url.searchParams;
  const archetype = searchParams.get('archetype') || 'The Scientist';
  const track = searchParams.get('track') || 'learn';
  
  try {
    // CRITICAL: Try Edge Config first (<5ms reads globally)
    const cacheKey = `lesson:${day}:meta`;
    const cached: LessonMetadata | null = await get(cacheKey);
    
    if (cached) {
      // Return cached metadata with optimized headers
      return new Response(
        JSON.stringify({
          ...cached,
          _source: 'edge-config',
          _cached: true,
        }),
        {
          status: 200,
          headers: {
            'Content-Type': 'application/json',
            'Cache-Control': 'public, s-maxage=3600, stale-while-revalidate=86400',
            'CDN-Cache-Control': 'public, s-maxage=3600',
            'X-Data-Source': 'edge-config',
          },
        }
      );
    }
    
    // Fallback to Supabase (slower, but works)
    const supabaseUrl = process.env.SUPABASE_URL;
    const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY;
    
    if (!supabaseUrl || !supabaseKey) {
      return new Response(
        JSON.stringify({ 
          error: 'Supabase configuration missing',
          _source: 'error'
        }),
        { status: 500, headers: { 'Content-Type': 'application/json' } }
      );
    }
    
    const supabase = createClient(supabaseUrl, supabaseKey);
    
    // Fetch lesson from Supabase
    const { data: lesson, error } = await supabase
      .from('core_lessons')
      .select(`
        day_number,
        topic,
        emoji,
        category,
        marketing_headline,
        headline,
        universal_truth,
        grow_track_id,
        lesson_atoms(archetype, phase)
      `)
      .eq('day_number', day)
      .single();
    
    if (error || !lesson) {
      return new Response(
        JSON.stringify({ 
          error: 'Lesson not found',
          _source: 'supabase-error'
        }),
        { status: 404, headers: { 'Content-Type': 'application/json' } }
      );
    }
    
    // Transform to lightweight metadata format
    const metadata: LessonMetadata = {
      day: lesson.day_number,
      topic: lesson.topic,
      emoji: lesson.emoji || '📚',
      category: lesson.category || '',
      headline: lesson.marketing_headline || lesson.headline || '',
      universal_truth: lesson.universal_truth,
      hasLearn: true,
      hasGrow: !!lesson.grow_track_id,
      phases: [...new Set((lesson.lesson_atoms || []).map((a: any) => a.phase))],
      archetypes: [...new Set((lesson.lesson_atoms || []).map((a: any) => a.archetype))],
    };
    
    // Return with shorter cache (not in Edge Config yet)
    return new Response(
      JSON.stringify({
        ...metadata,
        _source: 'supabase',
        _cached: false,
      }),
      {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'public, s-maxage=300',
          'X-Data-Source': 'supabase',
        },
      }
    );
      
  } catch (error: any) {
    console.error('[Edge Lesson API] Error:', error);
    return new Response(
      JSON.stringify({ 
        error: 'Internal server error',
        message: error.message,
        _source: 'error'
      }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}

