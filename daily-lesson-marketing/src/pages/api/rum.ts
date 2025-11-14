import type { APIRoute } from 'astro';

export const POST: APIRoute = async ({ request }) => {
  const enableRum = import.meta.env.ENABLE_RUM === 'true';

  if (!enableRum) {
    return new Response(
      JSON.stringify({ success: false, error: 'RUM endpoint disabled' }),
      { status: 403, headers: { 'Content-Type': 'application/json' } }
    );
  }

  try {
    const body = await request.json();
    const { lcp, cls, inp, url } = body;

    // Log metrics (in production, forward to analytics service)
    if (import.meta.env.DEV) {
      console.log('RUM metrics:', { lcp, cls, inp, url, timestamp: new Date().toISOString() });
    }

    return new Response(
      JSON.stringify({ success: true }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    );
  } catch (error: any) {
    return new Response(
      JSON.stringify({ success: false, error: error.message }),
      { status: 400, headers: { 'Content-Type': 'application/json' } }
    );
  }
};











