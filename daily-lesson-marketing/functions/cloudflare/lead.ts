// Cloudflare Workers adapter
import { handleLeadRequest } from '../handlers/lead';

export default {
  async fetch(request: Request, env: any): Promise<Response> {
    if (request.method !== 'POST') {
      return new Response(JSON.stringify({ error: 'Method not allowed' }), {
        status: 405,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const body = await request.json();
    const result = await handleLeadRequest({
      body,
      headers: Object.fromEntries(request.headers.entries())
    });

    return new Response(JSON.stringify(result.body), {
      status: result.status,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};











