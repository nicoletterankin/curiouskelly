/**
 * GET /api/lesson/:day/:phase
 * Serves lesson content, personalized by age/archetype/language
 * Implements fallback chain: video > audio > text
 * Supports A/B testing variants if available
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL!,
  process.env.SUPABASE_ANON_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY!
);

const PHASES = ['hook', 'q1', 'q2', 'q3', 'wisdom', 'welcome', 'socratic', 'reveal', 'explore', 'wonder', 'reflect'];
const PHASE_MAP: Record<string, string> = {
  'hook': 'Hook',
  'q1': 'Fact1',
  'q2': 'Fact2', 
  'q3': 'Fact3',
  'wisdom': 'Wisdom',
  'welcome': 'Welcome',
  'socratic': 'Socratic',
  'reveal': 'Reveal',
  'explore': 'Explore',
  'wonder': 'Wonder',
  'reflect': 'Reflect',
};

const STORAGE_BUCKET = 'lesson-audio';

// Media type hierarchy for fallback
type MediaType = 'video' | 'audio' | 'text';

interface MediaResult {
  type: MediaType;
  url: string | null;
  available: boolean;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  try {
    const { day, phase } = req.query;
    const dayNumber = parseInt(day as string);
    const phaseKey = (phase as string).toLowerCase();
    
    // Query params for personalization
    const age = req.query.age as string || 'adult';
    const archetype = req.query.archetype as string || 'The Scientist';
    const language = req.query.language as string || 'en';
    
    if (!dayNumber || dayNumber < 1 || dayNumber > 365) {
      return res.status(400).json({ error: 'Invalid day number (1-365)' });
    }
    
    if (!PHASES.includes(phaseKey)) {
      return res.status(400).json({ error: `Invalid phase. Use: ${PHASES.join(', ')}` });
    }
    
    // Get core lesson
    const { data: lesson, error: lessonError } = await supabase
      .from('core_lessons')
      .select('id, topic, universal_truth, icon_emoji')
      .eq('day_number', dayNumber)
      .single();
    
    if (lessonError || !lesson) {
      return res.status(404).json({ error: `Lesson not found for day ${dayNumber}` });
    }
    
    // Get lesson atom for archetype/phase
    const dbPhase = PHASE_MAP[phaseKey] || phaseKey;
    const { data: atom } = await supabase
      .from('lesson_atoms')
      .select('content')
      .eq('core_lesson_id', lesson.id)
      .eq('archetype', archetype)
      .eq('phase', dbPhase)
      .single();
    
    // === FALLBACK CHAIN: video > audio > text ===
    const mediaResults: Record<MediaType, MediaResult> = {
      video: { type: 'video', url: null, available: false },
      audio: { type: 'audio', url: null, available: false },
      text: { type: 'text', url: null, available: false },
    };

    // 1. Try to get video URL (highest priority)
    const { data: video } = await supabase
      .from('lesson_video_generation_status')
      .select('video_url')
      .eq('core_lesson_id', lesson.id)
      .eq('archetype', archetype)
      .eq('phase', dbPhase)
      .eq('status', 'completed')
      .single();
    
    if (video?.video_url) {
      mediaResults.video = { type: 'video', url: video.video_url, available: true };
    }

    // 2. Try to get audio URL (fallback if no video)
    const audioPath = `day-${String(dayNumber).padStart(3, '0')}/${language}/${archetype}/${phaseKey}.mp3`;
    
    try {
      const { data: audioFile } = await supabase.storage
        .from(STORAGE_BUCKET)
        .createSignedUrl(audioPath, 3600); // 1 hour expiry
      
      if (audioFile?.signedUrl) {
        mediaResults.audio = { type: 'audio', url: audioFile.signedUrl, available: true };
      }
    } catch (audioError) {
      // Audio not available - this is expected for ungenerated content
      console.log(`[Lesson] No audio for day ${dayNumber}, phase ${phaseKey}`);
    }

    // 3. Text is always available if we have content
    const content = atom?.content || {};
    
    // Get age-specific shard if available
    const ageMap: Record<string, number> = {
      'child': 9,
      'teen': 15,
      'young_adult': 26,
      'adult': 26,
      'elder': 72,
    };
    const targetAge = ageMap[age] || 26;
    
    const { data: shard } = await supabase
      .from('lesson_shards')
      .select('script_content')
      .eq('core_lesson_id', lesson.id)
      .eq('age', targetAge)
      .eq('region', language)
      .single();
    
    // Build text content
    const script = shard?.script_content?.script || content.script || content.text || '';
    const translations = content.translations?.[language];
    const finalScript = translations?.script || script;
    
    if (finalScript) {
      mediaResults.text = { type: 'text', url: null, available: true };
    }

    // Determine best available media type
    let bestMediaType: MediaType = 'text';
    let primaryUrl: string | null = null;
    
    if (mediaResults.video.available) {
      bestMediaType = 'video';
      primaryUrl = mediaResults.video.url;
    } else if (mediaResults.audio.available) {
      bestMediaType = 'audio';
      primaryUrl = mediaResults.audio.url;
    }

    // Build response with fallback information
    return res.status(200).json({
      day: dayNumber,
      phase: phaseKey,
      topic: lesson.topic,
      universal_truth: lesson.universal_truth,
      icon: lesson.icon_emoji,
      
      // Content
      script: finalScript,
      options: translations?.options || content.options || [],
      
      // Media - Primary (best available)
      media_type: bestMediaType,
      media_url: primaryUrl,
      
      // Media - Individual URLs for explicit access
      video_url: mediaResults.video.url,
      audio_url: mediaResults.audio.url,
      
      // Fallback information
      fallback_chain: {
        video: mediaResults.video.available,
        audio: mediaResults.audio.available,
        text: mediaResults.text.available,
        using: bestMediaType,
      },
      
      // Metadata
      archetype,
      age_bucket: age,
      language,
      
      // Kelly presentation
      kelly_pose: content.kellyPose || 'neutral',
      kelly_emotion: content.kellyEmotion || 'curious',
    });
    
  } catch (err) {
    console.error('Lesson handler error:', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
