/**
 * Simple ping endpoint to test Vercel deployment
 * Uses minimal syntax for maximum compatibility
 */

export default function handler(req: any, res: any) {
  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Origin', '*');
    return res.status(200).end();
  }
  
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Cache-Control', 'no-store');
  
  return res.status(200).send(JSON.stringify({ 
    ok: true, 
    ts: Date.now(),
    env: process.env.NODE_ENV || 'unknown'
  }));
}
