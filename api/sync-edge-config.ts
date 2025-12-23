/**
 * Sync Worker: Supabase → Edge Config
 * 
 * This endpoint syncs lesson metadata from Supabase to Edge Config
 * for instant global reads (<5ms).
 * 
 * Call this via webhook when lesson content updates in Supabase.
 */

import { set } from '@vercel/edge-config';
import { createClient } from '@supabase/supabase-js';

export const config = {
  runtime: 'edge',
};

export default async function handler(req: Request): Promise<Response> {

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
  // Verify secret for security
  let body: any = {};
  try {
    body = await req.json();
  } catch {
    // No body, continue
  }
  
  const { secret, day } = body;
  
  if (secret !== process.env.EDGE_CONFIG_SYNC_SECRET) {
    return new Response(
      JSON.stringify({ error: 'Unauthorized' }),
      { status: 401, headers: { 'Content-Type': 'application/json' } }
    );
  }
  
  const dayNum = day ? parseInt(day as string) : null;
  
  if (dayNum && (dayNum < 1 || dayNum > 365)) {
    return new Response(
      JSON.stringify({ error: 'Invalid day number (1-365)' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } }
    );
  }
  
  try {
    const supabaseUrl = process.env.SUPABASE_URL;
    const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
    
    if (!supabaseUrl || !supabaseKey) {
      return new Response(
        JSON.stringify({ error: 'Supabase configuration missing' }),
        { status: 500, headers: { 'Content-Type': 'application/json' } }
      );
    }
    
    const supabase = createClient(supabaseUrl, supabaseKey);
    
    // If specific day provided, sync that day only
    if (dayNum) {
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
        .eq('day_number', dayNum)
        .single();
      
      if (error || !lesson) {
        return new Response(
          JSON.stringify({ error: `Lesson ${dayNum} not found` }),
          { status: 404, headers: { 'Content-Type': 'application/json' } }
        );
      }
      
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
      
      // Store in Edge Config
      await set(`lesson:${dayNum}:meta`, metadata);
      
      return new Response(
        JSON.stringify({ 
          success: true, 
          day: dayNum,
          synced: true 
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      );
    }
    
    // If no day specified, sync all lessons (batch operation)
    const { data: lessons, error } = await supabase
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
      .order('day_number');
    
    if (error) {
      return new Response(
        JSON.stringify({ error: 'Failed to fetch lessons', details: error.message }),
        { status: 500, headers: { 'Content-Type': 'application/json' } }
      );
    }
    
    // Sync all lessons to Edge Config
    const syncPromises = lessons.map(async (lesson) => {
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
      
      return set(`lesson:${lesson.day_number}:meta`, metadata);
    });
    
    await Promise.all(syncPromises);
    
    return new Response(
      JSON.stringify({ 
        success: true, 
        synced: lessons.length,
        message: `Synced ${lessons.length} lessons to Edge Config`
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    );
    
  } catch (error: any) {
    console.error('[Sync Edge Config] Error:', error);
    return new Response(
      JSON.stringify({ 
        error: 'Internal server error',
        message: error.message 
      }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}

