// Vercel adapter
import { handleLeadRequest } from '../handlers/lead';

export default async function handler(req: any) {
  if (req.method !== 'POST') {
    return {
      statusCode: 405,
      body: JSON.stringify({ error: 'Method not allowed' })
    };
  }

  const result = await handleLeadRequest({
    body: JSON.parse(req.body || '{}'),
    headers: req.headers
  });

  return {
    statusCode: result.status,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(result.body)
  };
}











