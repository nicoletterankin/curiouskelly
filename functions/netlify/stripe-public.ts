/**
 * Stripe Public Config (GET /api/stripe-public)
 * Returns ONLY safe-to-expose values needed by Stripe.js (publishable key).
 */

type NetlifyHandlerEvent = {
  httpMethod: string;
  headers: Record<string, string>;
};

type NetlifyHandlerContext = {
  env: Record<string, string>;
};

type NetlifyHandler = (event: NetlifyHandlerEvent, context: NetlifyHandlerContext) => Promise<{
  statusCode: number;
  headers: Record<string, string>;
  body: string;
}>;

export const handler: NetlifyHandler = async (event, context) => {
  const baseHeaders: Record<string, string> = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
    'Content-Type': 'application/json',
    'Cache-Control': 'public, max-age=3600',
  };

  // Handle CORS preflight
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 204,
      headers: baseHeaders,
      body: ''
    };
  }

  if (event.httpMethod !== 'GET') {
    return {
      statusCode: 405,
      headers: baseHeaders,
      body: JSON.stringify({ error: 'method_not_allowed' })
    };
  }

  const publishableKey = process.env.STRIPE_PUBLISHABLE_KEY || context.env?.STRIPE_PUBLISHABLE_KEY;

  if (!publishableKey) {
    return {
      statusCode: 503,
      headers: baseHeaders,
      body: JSON.stringify({ 
        error: 'stripe_not_configured', 
        message: 'Missing STRIPE_PUBLISHABLE_KEY' 
      })
    };
  }

  return {
    statusCode: 200,
    headers: baseHeaders,
    body: JSON.stringify({ publishableKey })
  };
};

