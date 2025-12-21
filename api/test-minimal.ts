/**
 * Minimal test API - zero dependencies
 * GET /api/test-minimal
 */
export default function handler(req: any, res: any) {
  return res.status(200).json({ 
    ok: true, 
    time: new Date().toISOString(),
    message: 'API is working!'
  });
}

