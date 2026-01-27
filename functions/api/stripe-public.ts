/**
 * Stripe Public Config (GET /api/stripe-public)
 * Returns ONLY safe-to-expose values needed by Stripe.js (publishable key).
 */

type CloudflareContext = {
  request: Request;
  env: Record<string, string | undefined>;
};

export const onRequestGet = async (context: CloudflareContext) => {
  const publishableKey = context.env.STRIPE_PUBLISHABLE_KEY;
  
  if (!publishableKey) {
    return new Response(
      JSON.stringify({ error: 'stripe_not_configured', message: 'Missing STRIPE_PUBLISHABLE_KEY' }),
      { 
        status: 503, 
        headers: { 
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        } 
      }
    );
  }

  return new Response(
    JSON.stringify({ publishableKey }),
    { 
      status: 200, 
      headers: { 
        'Content-Type': 'application/json',
        'Cache-Control': 'public, max-age=3600',
        'Access-Control-Allow-Origin': '*'
      } 
    }
  );
};

// Handle CORS preflight
export const onRequestOptions = async () => {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Access-Control-Max-Age': '86400'
    }
  });
};





