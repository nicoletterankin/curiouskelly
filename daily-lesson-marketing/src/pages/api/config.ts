
import type { APIRoute } from 'astro';

export const GET: APIRoute = async () => {
  return new Response(
    JSON.stringify({
      supabaseUrl: import.meta.env.PUBLIC_SUPABASE_URL || process.env.PUBLIC_SUPABASE_URL,
      supabaseKey: import.meta.env.PUBLIC_SUPABASE_ANON_KEY || process.env.PUBLIC_SUPABASE_ANON_KEY,
    }),
    {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );
};







