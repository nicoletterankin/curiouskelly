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
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    // Run analysis RPCs
    const { data: impacts, error: impactsError } = await supabase.rpc('calculate_all_impacts')
    const { data: incoherencies, error: incoherenciesError } = await supabase.rpc('detect_incoherencies')
    const { data: improvements, error: improvementsError } = await supabase.rpc('generate_improvements')

    return new Response(JSON.stringify({
      success: true,
      timestamp: new Date().toISOString(),
      results: {
        impacts_updated: impacts?.[0]?.updated_count || 0,
        incoherencies_detected: incoherencies?.[0]?.detected || 0,
        improvements_queued: improvements?.[0]?.queued || 0,
      },
      errors: {
        impacts: impactsError?.message,
        incoherencies: incoherenciesError?.message,
        improvements: improvementsError?.message,
      }
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
