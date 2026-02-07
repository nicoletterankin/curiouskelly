'use client'

import { createBrowserClient } from '@supabase/ssr'

// Singleton pattern to prevent multiple instances
let supabaseClient: ReturnType<typeof createBrowserClient> | null = null

export function getSupabaseBrowserClient() {
  if (supabaseClient) return supabaseClient
  
  supabaseClient = createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY!
  )
  
  return supabaseClient
}

// Hook for use in components
export function useSupabase() {
  return getSupabaseBrowserClient()
}
