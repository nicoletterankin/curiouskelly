import type { VercelRequest, VercelResponse } from '@vercel/node';

export default function handler(req: VercelRequest, res: VercelResponse) {
  res.status(200).json({ 
    status: 'ok',
    timestamp: new Date().toISOString(),
    node: process.version,
    env: {
      hasStripeKey: !!process.env.STRIPE_SECRET_KEY,
      hasStripePrice: !!process.env.STRIPE_PRICE_MONTHLY,
      hasElevenLabsKey: !!process.env.ELEVENLABS_API_KEY,
      hasElevenLabsVoice: !!process.env.ELEVENLABS_VOICE_ID,
      hasSupabaseUrl: !!process.env.PUBLIC_SUPABASE_URL,
      hasSupabaseKey: !!process.env.SUPABASE_SERVICE_ROLE_KEY,
      hasReplicateToken: !!process.env.REPLICATE_API_TOKEN
    }
  });
}


