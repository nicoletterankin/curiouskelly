import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const url = new URL(req.url)
    const day = parseInt(url.searchParams.get('day') || '1')
    const phase = url.searchParams.get('phase') || 'hook'
    const age = parseInt(url.searchParams.get('age') || '26')
    const archetype = url.searchParams.get('archetype') || 'The Explorer'
    const language = url.searchParams.get('language') || 'en'

    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? ''
    )

    // Get core lesson
    const { data: lesson } = await supabase
      .from('core_lessons')
      .select('id, topic, day_number')
      .eq('day_number', day)
      .single()

    if (!lesson) {
      return new Response(JSON.stringify({ error: 'Lesson not found' }), {
        status: 404,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      })
    }

    // Get archetype-specific content
    const { data: atom } = await supabase
      .from('lesson_atoms')
      .select('content')
      .eq('core_lesson_id', lesson.id)
      .eq('archetype', archetype)
      .eq('phase', phase.charAt(0).toUpperCase() + phase.slice(1))
      .single()

    // Get age-specific shard
    const { data: shard } = await supabase
      .from('lesson_shards')
      .select('script_content')
      .eq('core_lesson_id', lesson.id)
      .eq('age', age)
      .single()

    // Get video if available
    const { data: video } = await supabase
      .from('lesson_video_generation_status')
      .select('video_url')
      .eq('day_of_year', day)
      .eq('phase', phase)
      .eq('status', 'completed')
      .single()

    return new Response(JSON.stringify({
      day,
      phase,
      topic: lesson.topic,
      script: atom?.content?.script || shard?.script_content?.phases?.[phase]?.script_en || '',
      video_url: video?.video_url || null,
      archetype,
      age,
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    })
  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    })
  }
})
