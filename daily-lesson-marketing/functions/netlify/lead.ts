// Netlify adapter
import { handleLeadRequest } from '../handlers/lead';

export const handler = async (event: any, context: any) => {
  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      body: JSON.stringify({ error: 'Method not allowed' })
    };
  }

  const result = await handleLeadRequest({
    body: JSON.parse(event.body || '{}'),
    headers: event.headers
  });

  return {
    statusCode: result.status,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(result.body)
  };
};










