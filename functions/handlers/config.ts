import type { Request, Response } from '@vercel/node';

export async function configHandler(
  request: Request,
  context: { env: Record<string, string | undefined> }
): Promise<Response> {
  // Only expose public keys - never expose secret keys!
  return new Response(
    JSON.stringify({
      supabaseUrl: context.env.PUBLIC_SUPABASE_URL || 'https://tvjalxxsyryjphkforjv.supabase.co',
      supabaseAnonKey: context.env.PUBLIC_SUPABASE_ANON_KEY || '',
      // Don't expose secret keys like ELEVENLABS_API_KEY or STRIPE_SECRET_KEY
    }),
    {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'public, max-age=3600', // Cache for 1 hour
      },
    }
  );
}

