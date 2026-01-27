import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

// In-memory session tracking (use Redis in production)
const sessions = new Map<string, { startTime: number; lastPing: number; lessonDay: number; phase: string }>()

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { action, session_id, student_id, lesson_day, phase } = await req.json()

    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    const now = Date.now()

    switch (action) {
      case 'start':
        sessions.set(session_id, {
          startTime: now,
          lastPing: now,
          lessonDay: lesson_day,
          phase,
        })
        return new Response(JSON.stringify({ success: true, session_id }), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        })

      case 'ping':
        const session = sessions.get(session_id)
        if (session) {
          session.lastPing = now
        }
        return new Response(JSON.stringify({ success: true, alive: !!session }), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        })

      case 'end':
        const endSession = sessions.get(session_id)
        if (endSession) {
          const duration = Math.floor((now - endSession.startTime) / 1000)
          sessions.delete(session_id)
          
          // Record the time
          await supabase
            .from('student_feedback')
            .insert({
              lesson_day: endSession.lessonDay,
              phase: endSession.phase,
              student_id,
              time_on_phase_seconds: duration,
              completed: duration > 60, // Consider completed if > 60s
            })

          return new Response(JSON.stringify({ success: true, duration_seconds: duration }), {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          })
        }
        return new Response(JSON.stringify({ success: false, error: 'Session not found' }), {
          status: 404,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        })

      default:
        return new Response(JSON.stringify({ error: 'Invalid action' }), {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        })
    }
  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    })
  }
})
