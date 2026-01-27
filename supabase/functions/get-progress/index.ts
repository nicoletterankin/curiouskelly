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
    const studentId = url.searchParams.get('student_id')

    if (!studentId) {
      return new Response(JSON.stringify({ error: 'Missing student_id' }), {
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      })
    }

    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? ''
    )

    // Get completed lessons
    const { data: feedback } = await supabase
      .from('student_feedback')
      .select('lesson_day, phase, completed')
      .eq('student_id', studentId)
      .eq('completed', true)

    // Calculate completed days
    const phasesByDay = new Map<number, Set<string>>()
    feedback?.forEach((f: any) => {
      if (!phasesByDay.has(f.lesson_day)) {
        phasesByDay.set(f.lesson_day, new Set())
      }
      phasesByDay.get(f.lesson_day)!.add(f.phase)
    })

    const REQUIRED_PHASES = ['hook', 'q1', 'q2', 'q3', 'wisdom']
    const completedDays: number[] = []
    
    phasesByDay.forEach((phases, day) => {
      if (REQUIRED_PHASES.every(p => phases.has(p))) {
        completedDays.push(day)
      }
    })

    // Calculate streak
    const today = new Date()
    const dayOfYear = Math.ceil((today.getTime() - new Date(today.getFullYear(), 0, 1).getTime()) / (1000 * 60 * 60 * 24))
    
    let streak = 0
    const sortedDays = [...completedDays].sort((a, b) => b - a)
    
    for (let i = 0; i < sortedDays.length; i++) {
      if (sortedDays[i] === dayOfYear - i) {
        streak++
      } else {
        break
      }
    }

    return new Response(JSON.stringify({
      student_id: studentId,
      completed_days: completedDays.length,
      completed_days_list: completedDays.sort((a, b) => a - b),
      total_days: 365,
      progress_percent: Math.round((completedDays.length / 365) * 1000) / 10,
      current_streak: streak,
      next_lesson: Math.min((sortedDays[0] || 0) + 1, 365),
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
